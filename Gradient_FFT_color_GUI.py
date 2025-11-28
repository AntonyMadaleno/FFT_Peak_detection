# -*- coding: utf-8 -*-
"""
FFT Filter POC (GUI) — GPU/CPU backend, color support, quaternion + FVG

Features:
- Load image with OpenCV (if available) or matplotlib
- Display image in color, grayscale or single band
- Choose FFT mode for peak detection: luminance, per-channel marginal, quaternion (vector) FFT magnitude
- Adaptive peak detection (with optional fast downsampling)
- Remove conjugates, merge similar sigmas
- Directional derivative-of-Gaussian kernels, compute gradients
- For color modes compute Full Vector Gradient (FVG) and optionally show per-channel responses

This file is meant to be edited in the Canvas. Comments in English, UTF-8.
"""

import math, time
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ------------------- utilities ------------------------------------------------

def try_imread(path):
    """Load image using cv2 if available, otherwise matplotlib. Return RGB float in [0,1]."""
    try:
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError('cv2.imread returned None')
        # if 16-bit or other, convert to float
        if bgr.dtype == np.uint8:
            arr = bgr.astype(np.float32) / 255.0
        else:
            # normalize by max
            arr = bgr.astype(np.float32)
            arr = arr / (arr.max() if arr.max()>0 else 1.0)
        # BGR -> RGB
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3][..., ::-1]
        return arr
    except Exception:
        # fallback
        import matplotlib.image as mpimg
        arr = mpimg.imread(path)
        # if integer dtype, normalize
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[..., :3]
        if arr.ndim == 2:
            return arr
        return arr


def ensure_gray(image):
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        gray = 0.2989*arr[...,0] + 0.5870*arr[...,1] + 0.1140*arr[...,2]
    else:
        gray = arr
    gray = gray - gray.min()
    if gray.max() != 0:
        gray = gray / gray.max()
    return gray


def downsample_block_mean(img, factor):
    if factor <= 1:
        return img
    arr = np.asarray(img)
    if arr.ndim == 2:
        H, W = arr.shape
        H2 = H // factor; W2 = W // factor
        arr = arr[:H2*factor, :W2*factor]
        arr = arr.reshape(H2, factor, W2, factor).mean(axis=(1,3))
        return arr
    elif arr.ndim == 3:
        H, W, C = arr.shape
        H2 = H // factor; W2 = W // factor
        arr = arr[:H2*factor, :W2*factor, :]
        arr = arr.reshape(H2, factor, W2, factor, C).mean(axis=(1,3))
        return arr
    else:
        raise ValueError('Unsupported image dimensions for downsample_block_mean')

# ------------------- FFT / kernels / FVG -------------------------------------

def compute_fft_shift_mag_np(img):
    F = np.fft.fft2(img)
    Fsh = np.fft.fftshift(F)
    mag = np.abs(Fsh)
    mag_log = np.log1p(mag)
    return Fsh, mag, mag_log


def fft_convolve2d_backend(xp, img, kernel):
    # xp: np or cp module
    if xp is np:
        H, W = img.shape
        kh, kw = kernel.shape
        pad_h = H + kh - 1; pad_w = W + kw - 1
        IMG = np.fft.fft2(img, s=(pad_h, pad_w))
        KER = np.fft.fft2(kernel, s=(pad_h, pad_w))
        RES = IMG * KER
        out = np.fft.ifft2(RES).real
        out = out[kh//2:kh//2+H, kw//2:kw//2+W]
        return out
    else:
        import cupy as cp
        H, W = img.shape
        kh, kw = kernel.shape
        pad_h = H + kh - 1; pad_w = W + kw - 1
        IMG = cp.fft.fft2(img, s=(pad_h, pad_w))
        KER = cp.fft.fft2(kernel, s=(pad_h, pad_w))
        RES = IMG * KER
        out = cp.fft.ifft2(RES).real
        out = out[kh//2:kh//2+H, kw//2:kw//2+W]
        return out


def directional_gaussian_derivative_kernel(size, sigma, theta, xp=np):
    # size: odd integer
    half = size//2
    ax = xp.arange(-half, half+1)
    xx, yy = xp.meshgrid(ax, ax)
    x_rot = xx * xp.cos(theta) + yy * xp.sin(theta)
    y_rot = -xx * xp.sin(theta) + yy * xp.cos(theta)
    g = xp.exp(-(x_rot**2 + y_rot**2) / (2.0 * sigma**2))
    dg = -x_rot / (sigma**2) * g
    # normalize absolute sum
    s = float(xp.sum(xp.abs(dg)))
    if s != 0:
        dg = dg / s
    return dg


def quaternion_transform_rgb(img_rgb, scalar_luma=True):
    arr = np.asarray(img_rgb, dtype=np.float64)
    r = arr[...,0]; g = arr[...,1]; b = arr[...,2]
    if scalar_luma:
        s = 0.2989*r + 0.5870*g + 0.1140*b
    else:
        s = np.zeros_like(r)
    vec = np.stack([r,g,b], axis=-1)
    return s, vec


def compute_fft_vector_magnitude(img_rgb_small):
    # compute FFT shift for each channel and return sqrt(sum(|Fch|^2))
    ch_mags = []
    for ch in range(3):
        Fsh = np.fft.fftshift(np.fft.fft2(img_rgb_small[...,ch]))
        ch_mags.append(np.abs(Fsh))
    arr = np.stack(ch_mags, axis=0)
    mag = np.sqrt(np.sum(arr**2, axis=0))
    mag_log = np.log1p(mag)
    return mag, mag_log


def compute_fvg(kernel, img_rgb, backend=np):
    # backend: np or cp module
    H,W,C = img_rgb.shape
    per_ch = []
    for ch in range(3):
        ch_img = img_rgb[...,ch].astype(np.float64)
        if backend is np:
            res = fft_convolve2d_backend(np, ch_img, kernel)
        else:
            import cupy as cp
            ch_b = cp.asarray(ch_img); ker_b = cp.asarray(kernel)
            res_b = fft_convolve2d_backend(cp, ch_b, ker_b)
            res = cp.asnumpy(res_b)
        per_ch.append(res)
    per_ch = [np.asarray(x, dtype=np.float64) for x in per_ch]
    sq = np.zeros_like(per_ch[0])
    for r in per_ch:
        sq += r**2
    fvg = np.sqrt(sq)
    return fvg, per_ch

# ------------------- GUI -----------------------------------------------------

class FFTFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FFT Directional Filters - Color & Quaternion FVG')
        self.geometry('1400x920')
        self.img_cpu = None
        self.gray = None
        self.peaks_info = []
        self.kernels = []
        self.gradients = []
        self.per_channel = []
        self._build_controls()
        self._build_display()

    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(side='top', fill='x', padx=6, pady=6)
        ttk.Button(frame, text='Load Image', command=self._on_load_image).grid(row=0, column=0, padx=4)
        ttk.Label(frame, text='Device:').grid(row=0, column=1, padx=4)
        self.device_var = tk.StringVar(value='auto')
        ttk.Combobox(frame, textvariable=self.device_var, values=('auto','cpu','gpu'), width=6).grid(row=0, column=2, padx=4)
        self.fast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text='Fast detection', variable=self.fast_var).grid(row=0, column=3, padx=10)
        ttk.Label(frame, text='Adaptive k:').grid(row=0, column=4, padx=4)
        self.k_var = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.k_var, width=6).grid(row=0, column=5, padx=4)
        ttk.Label(frame, text='min_distance:').grid(row=0, column=6, padx=4)
        self.min_distance_var = tk.IntVar(value=10)
        tk.Spinbox(frame, from_=1, to=200, textvariable=self.min_distance_var, width=6).grid(row=0, column=7, padx=4)
        self.timer_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text='Timer', variable=self.timer_var).grid(row=0, column=8, padx=8)

        ttk.Label(frame, text='Display mode:').grid(row=1, column=0, padx=4)
        self.display_var = tk.StringVar(value='color')
        ttk.Combobox(frame, textvariable=self.display_var, values=('color','grayscale','band 0','band 1','band 2'), width=10).grid(row=1, column=1, padx=4)

        ttk.Label(frame, text='FFT mode:').grid(row=1, column=2, padx=4)
        self.fft_mode_var = tk.StringVar(value='luminance')
        ttk.Combobox(frame, textvariable=self.fft_mode_var, values=('luminance','per-channel-marginal','quaternion'), width=18).grid(row=1, column=3, padx=4)

        ttk.Label(frame, text='Color mode:').grid(row=1, column=4, padx=4)
        self.color_mode_var = tk.StringVar(value='grayscale')
        ttk.Combobox(frame, textvariable=self.color_mode_var, values=('grayscale','per-channel','quaternion+FVG'), width=15).grid(row=1, column=5, padx=4)

        ttk.Button(frame, text='Detect Peaks', command=self._on_detect).grid(row=1, column=6, padx=6)

        # results tree
        self.filter_tree = ttk.Treeview(frame, columns=('pos','sigma','angle'), show='headings', height=6)
        for col, txt, w in [('pos','pos (r,c)',120),('sigma','sigma (px)',110),('angle','angle (deg)',110)]:
            self.filter_tree.heading(col, text=txt)
            self.filter_tree.column(col, width=w, anchor='center')
        self.filter_tree.grid(row=2, column=0, columnspan=8, sticky='w')
        self.filter_tree.bind('<<TreeviewSelect>>', self._on_select_filter)

        self.timing_label = ttk.Label(frame, text='')
        self.timing_label.grid(row=2, column=8, columnspan=4, sticky='w')

    def _build_display(self):
        fig = Figure(figsize=(11,7))
        self.ax_img = fig.add_subplot(231)
        self.ax_fft = fig.add_subplot(232)
        self.ax_fft_roi = fig.add_subplot(233)
        self.ax_kernel = fig.add_subplot(234)
        self.ax_grad = fig.add_subplot(235)
        self.ax_channels = fig.add_subplot(236)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _on_load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files','*.png;*.jpg;*.jpeg;*.bmp;*.tif')])
        if not path: return
        arr = try_imread(path)
        if arr.ndim == 2:
            self.img_cpu = np.stack([arr,arr,arr], axis=-1)
        else:
            self.img_cpu = np.asarray(arr, dtype=np.float64)
        # normalize
        self.img_cpu = np.clip(self.img_cpu, 0.0, 1.0)
        self.gray = ensure_gray(self.img_cpu)
        self._draw_image()

    def _draw_image(self):
        if self.img_cpu is None: return
        mode = self.display_var.get()
        self.ax_img.clear()
        if mode == 'color':
            self.ax_img.imshow(self.img_cpu)
        elif mode == 'grayscale':
            self.ax_img.imshow(self.gray, cmap='gray')
        elif mode.startswith('band'):
            b = int(mode.split()[-1])
            self.ax_img.imshow(self.img_cpu[..., b], cmap='gray')
        self.ax_img.set_title('Input image')
        self.ax_img.axis('off')
        self.canvas.draw()

    def _on_detect(self):
        if self.img_cpu is None:
            messagebox.showerror('Error','Load an image first'); return
        start = time.time()
        device_choice = self.device_var.get()
        # backend selection (CPU only here for simplicity; CuPy can be added)
        backend = np
        is_gpu = False

        # prepare small/full images
        color_mode = self.color_mode_var.get()
        fft_mode = self.fft_mode_var.get()
        img = self.img_cpu
        gray = self.gray

        # choose downsample for detection
        down_factor = 1
        maxdim = max(gray.shape)
        if self.fast_var.get() and maxdim > 1024:
            target = 512
            down_factor = math.ceil(maxdim / target)

        if down_factor > 1:
            if img.ndim == 3:
                img_small = downsample_block_mean(img, down_factor)
                gray_small = ensure_gray(img_small)
            else:
                gray_small = downsample_block_mean(gray, down_factor)
                img_small = None
        else:
            gray_small = gray
            img_small = img if img.ndim==3 else None

        # compute FFT magnitude depending on fft_mode
        if fft_mode == 'luminance':
            _, mag_small, mag_small_log = compute_fft_shift_mag_np(gray_small)
        elif fft_mode == 'per-channel-marginal':
            mag_small, mag_small_log = compute_fft_vector_magnitude(img_small if img_small is not None else np.stack([gray_small]*3,axis=-1))
        else:  # quaternion: treat as vector FFT magnitude (approximate)
            # compute FFT magnitude from RGB vector (same as per-channel vector) for now
            mag_small, mag_small_log = compute_fft_vector_magnitude(img_small if img_small is not None else np.stack([gray_small]*3,axis=-1))

        # adaptive threshold
        k = float(self.k_var.get())
        flat = mag_small.ravel(); med = np.median(flat); mad = np.median(np.abs(flat-med))
        robust_std = 1.4826*mad if mad>0 else flat.std()
        thresh = med + k * robust_std

        # find local maxima candidates
        def local_maxima_candidates(mag, threshold, min_distance=8, exclude_radius=6):
            ny, nx = mag.shape
            candidates = []
            for r in range(1, ny-1):
                row = mag[r]
                if row.max() < threshold: continue
                for c in range(1, nx-1):
                    val = mag[r,c]
                    if val < threshold: continue
                    neigh = mag[r-1:r+2, c-1:c+2]
                    if val >= neigh.max():
                        candidates.append((r,c,val))
            candidates.sort(key=lambda x: x[2], reverse=True)
            kept = []
            for (r,c,val) in candidates:
                too_close = False
                for (kr,kc,_) in kept:
                    if (r-kr)**2 + (c-kc)**2 <= (min_distance**2):
                        too_close = True; break
                if not too_close: kept.append((r,c,val))
            cy, cx = mag.shape[0]//2, mag.shape[1]//2
            kept = [p for p in kept if (p[0]-cy)**2 + (p[1]-cx)**2 > (exclude_radius**2)]
            return kept

        min_dist_local = int(self.min_distance_var.get())
        candidates_small = local_maxima_candidates(mag_small, threshold=thresh, min_distance=max(4, min_dist_local//max(1,down_factor)))

        # remove conjugates
        ny_s, nx_s = mag_small.shape
        cy_s, cx_s = ny_s//2, nx_s//2
        cand_map = {(r,c):v for (r,c,v) in candidates_small}
        picked = []
        used = set()
        for (r,c,val) in candidates_small:
            if (r,c) in used: continue
            dr, dc = r-cy_s, c-cx_s
            r2, c2 = cy_s - dr, cx_s - dc
            if (r2,c2) in cand_map and (r2,c2) not in used:
                val2 = cand_map[(r2,c2)]
                if val >= val2:
                    picked.append((r,c,val)); used.add((r,c)); used.add((r2,c2))
                else:
                    picked.append((r2,c2,val2)); used.add((r,c)); used.add((r2,c2))
            else:
                picked.append((r,c,val)); used.add((r,c))
            if len(picked) >= 20: break

        if not picked:
            flat_idx = np.unravel_index(np.argsort(mag_small.ravel())[::-1], mag_small.shape)
            for i in range(len(flat_idx[0])):
                r,c = flat_idx[0][i], flat_idx[1][i]
                if (r-cy_s)**2 + (c-cx_s)**2 <= 6**2: continue
                picked.append((r,c,mag_small[r,c]))
                if len(picked) >= 20: break

        # map to full coords and compute sigma/angle
        ny_f, nx_f = gray.shape
        cy_f, cx_f = ny_f//2, nx_f//2
        peaks_info = []
        for (r_s, c_s, val_s) in picked:
            du = (c_s - cx_s) / float(nx_s)
            dv = (r_s - cy_s) / float(ny_s)
            f = math.hypot(du, dv)
            if f == 0: continue
            wavelength = 1.0 / f
            sigma = wavelength / (2.0 * math.pi)
            angle = math.atan2(dv, du)
            c_full = int(round(cx_f + du * nx_f))
            r_full = int(round(cy_f + dv * ny_f))
            peaks_info.append({'r':r_full, 'c':c_full, 'val':float(val_s), 'wavelength':wavelength, 'sigma':sigma, 'angle':angle, 'du':du, 'dv':dv})

        # merge similar sigmas
        peaks_info = sorted(peaks_info, key=lambda x: x['val'], reverse=True)
        kept = []
        tol = 1.25
        for p in peaks_info:
            similar = False
            for q in kept:
                r1 = p['sigma'] / q['sigma'] if q['sigma']>0 else float('inf')
                r2 = q['sigma'] / p['sigma'] if p['sigma']>0 else float('inf')
                if r1 <= tol and r2 <= tol:
                    similar = True; break
            if not similar:
                kept.append(p)
            if len(kept) >= 12: break
        peaks_info = kept

        # build kernels and gradients
        kernels = []
        grads = []
        per_ch_all = []
        for p in peaks_info:
            sigma = p['sigma']; angle = p['angle']
            size = max(3, int(math.ceil(6*sigma)))
            if size % 2 == 0: size += 1
            kernel = directional_gaussian_derivative_kernel(size, sigma, angle, xp=np)
            kernels.append(kernel)
            if img.ndim == 3 and color_mode in ('per-channel','quaternion+FVG'):
                fvg_map, per_ch = compute_fvg(kernel, img, backend=np)
                grads.append(fvg_map); per_ch_all.append(per_ch)
            else:
                grad = np.abs(fft_convolve2d_backend(np, gray, kernel))
                grads.append(grad); per_ch_all.append(None)

        # store and display
        self.peaks_info = peaks_info
        self.kernels = kernels
        self.gradients = grads
        self.per_channel = per_ch_all

        # full FFT for visualization
        if fft_mode == 'luminance':
            _, mag_full, mag_full_log = compute_fft_shift_mag_np(gray)
        else:
            # vector magnitude
            mag_full, mag_full_log = compute_fft_vector_magnitude(img if img.ndim==3 else np.stack([gray]*3,axis=-1))

        self.ax_img.clear()
        # display input image according to display mode
        self._draw_image()

        self.ax_fft.clear(); self.ax_fft.imshow(mag_full_log, cmap='inferno'); self.ax_fft.set_title('FFT magnitude (log)')
        for p in peaks_info:
            self.ax_fft.plot(p['c'], p['r'], 'go')
        self.ax_fft.axis('off')

        # ROI viewer: first peak
        self.ax_fft_roi.clear()
        if peaks_info:
            p0 = peaks_info[0]
            h = 60
            r0 = max(0, p0['r']-h); r1 = min(mag_full_log.shape[0], p0['r']+h+1)
            c0 = max(0, p0['c']-h); c1 = min(mag_full_log.shape[1], p0['c']+h+1)
            roi = mag_full_log[r0:r1, c0:c1]
            self.ax_fft_roi.imshow(roi, cmap='inferno'); self.ax_fft_roi.set_title('FFT ROI (first peak)'); self.ax_fft_roi.axis('off')
        else:
            self.ax_fft_roi.set_title('FFT ROI')
        self.canvas.draw()

        # fill tree
        for item in self.filter_tree.get_children(): self.filter_tree.delete(item)
        for p in peaks_info:
            self.filter_tree.insert('',tk.END,values=(f"({p['r']},{p['c']})",f"{p['sigma']:.2f}",f"{math.degrees(p['angle']):.1f}"))

        self.timing_label.config(text=f"Detect: {time.time()-start:.3f}s | peaks={len(peaks_info)}")

    def _on_select_filter(self, evt):
        sel = self.filter_tree.selection()
        if not sel: return
        idx = self.filter_tree.index(sel[0])
        self._show_filter(idx)

    def _show_filter(self, idx):
        if idx >= len(self.kernels): return
        kernel = self.kernels[idx]
        grad = self.gradients[idx]
        perch = self.per_channel[idx]
        self.ax_kernel.clear(); self.ax_kernel.imshow(kernel, cmap='seismic'); self.ax_kernel.set_title('Kernel (signed)'); self.ax_kernel.axis('off')
        self.ax_grad.clear(); self.ax_grad.imshow(grad, cmap='viridis'); self.ax_grad.set_title('Gradient (FVG or mag)'); self.ax_grad.axis('off')
        # show per-channel maps if available
        self.ax_channels.clear()
        if perch is not None:
            # perch is list of 3 arrays
            stack = np.stack([np.abs(perch[ch]) for ch in range(3)], axis=-1)
            # normalize for display
            mi = stack.min(); ma = stack.max()
            if ma>mi:
                stack = (stack - mi) / (ma - mi)
            self.ax_channels.imshow(stack); self.ax_channels.set_title('Per-channel responses (RGB)')
        else:
            self.ax_channels.set_title('Per-channel responses (N/A)')
            self.ax_channels.imshow(np.zeros((2,2)), cmap='gray')
        self.canvas.draw()

# ------------------- run -----------------------------------------------------
if __name__ == '__main__':
    app = FFTFilterGUI()
    app.mainloop()
