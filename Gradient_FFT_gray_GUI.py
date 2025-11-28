# -*- coding: utf-8 -*-
"""
POC GUI: FFT-based directional gaussian-derivative filters with optional GPU (CuPy) and Tkinter UI

Features:
- Load grayscale image (or auto-convert from color)
- Compute 2D FFT (shifted) and show log-magnitude
- Detect spectral peaks with adaptive threshold, remove conjugates, merge similar sigmas
- Build directional derivative-of-Gaussian kernels from detected peaks
- Convolve (FFT-based) and show gradient maps (cmap='viridis')
- GUI to pick a detected filter, show the kernel, gradient map, and the FFT point that selected it
- Option to force device: Auto / CPU / GPU
- Timing option (timer=True) to measure major blocks
- Acceleration option for large images: FFT downsampling (block-averaging) to quickly find candidate peaks

How to use:
  - Run the script: `python poc_gui_fft_filters.py`
  - Use the GUI to load an image, choose device, optionally enable "fast detection" for large images,
    then click "Detect Peaks". Select a detected filter from the list to visualize kernel/gradient/FFT point.
"""

import os
import time
import math
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------------- Backend selection (NumPy / CuPy) ----------------

def get_backend(use_gpu_choice=None):
    """
    Return (xp, is_gpu, cupy_available).
    use_gpu_choice: 'auto'|'cpu'|'gpu'
    """
    try:
        import cupy as cp
        cupy_available = True
    except Exception:
        cp = None
        cupy_available = False

    if use_gpu_choice == 'gpu':
        if not cupy_available:
            raise RuntimeError('CuPy requested but not available')
        return cp, True, True
    if use_gpu_choice == 'cpu':
        return np, False, cupy_available
    # auto
    if cupy_available:
        return cp, True, True
    return np, False, cupy_available


def to_cpu(xp, arr):
    if xp is np:
        return arr
    else:
        return xp.asnumpy(arr)


def sync_if_gpu(xp):
    if xp is not np:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

# ---------------- Utilities ----------------

def ensure_gray(image):
    """Accept numpy array (H,W) or (H,W,3/4) and return float64 grayscale normalized [0,1]."""
    arr = np.array(image, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr - arr.min()
    if arr.max() != 0:
        arr = arr / arr.max()
    return arr


def downsample_block_mean(img, factor):
    """Downsample image by integer factor with simple block mean. factor must be >=1."""
    if factor <= 1:
        return img
    ny, nx = img.shape
    ny2 = ny // factor
    nx2 = nx // factor
    img = img[:ny2*factor, :nx2*factor]
    # reshape and mean
    img_ds = img.reshape(ny2, factor, nx2, factor).mean(axis=(1,3))
    return img_ds

# ---------------- FFT / detection / kernels ----------------

def compute_fft_shift_mag(xp, img):
    F = xp.fft.fft2(img)
    Fsh = xp.fft.fftshift(F)
    mag = xp.abs(Fsh)
    mag_log = xp.log1p(mag)
    return Fsh, mag, mag_log


def adaptive_threshold_np(mag_np, k=4.0):
    flat = mag_np.ravel()
    med = np.median(flat)
    mad = np.median(np.abs(flat - med))
    robust_std = 1.4826 * mad if mad > 0 else flat.std()
    thresh = med + k * robust_std
    return float(thresh), float(med), float(robust_std)


def local_maxima_candidates_np(mag_np, threshold, min_distance=8):
    ny, nx = mag_np.shape
    candidates = []
    for r in range(1, ny-1):
        row = mag_np[r]
        if row.max() < threshold:
            continue
        for c in range(1, nx-1):
            val = mag_np[r, c]
            if val < threshold:
                continue
            neigh = mag_np[r-1:r+2, c-1:c+2]
            if val >= neigh.max():
                candidates.append((r, c, val))
    candidates.sort(key=lambda x: x[2], reverse=True)
    kept = []
    for (r, c, val) in candidates:
        too_close = False
        for (kr, kc, _) in kept:
            if (r-kr)**2 + (c-kc)**2 <= (min_distance**2):
                too_close = True
                break
        if not too_close:
            kept.append((r, c, val))
    return kept


def mirror_coord(coord, shape):
    r, c = coord
    ny, nx = shape
    cy, cx = ny//2, nx//2
    dr, dc = r - cy, c - cx
    r2 = cy - dr
    c2 = cx - dc
    return (r2, c2)


def directional_gaussian_derivative_kernel(sigma, angle_rad, truncate=3.0):
    if sigma <= 0:
        sigma = 0.5
    radius = int(math.ceil(truncate * sigma))
    ys = np.arange(-radius, radius+1)
    xs = np.arange(-radius, radius+1)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    ct = math.cos(angle_rad); st = math.sin(angle_rad)
    xprime = ct * xx + st * yy
    yprime = -st * xx + ct * yy
    rsq = xprime**2 + yprime**2
    g = np.exp(-rsq/(2*sigma*sigma))
    dgdxp = - (xprime / (sigma*sigma)) * g
    s = np.sum(np.abs(dgdxp))
    if s != 0.0:
        dgdxp = dgdxp / s
    return dgdxp

# FFT convolution on a backend (xp = np or cupy)
def fft_convolve2d_backend(xp, img, kernel):
    ny, nx = img.shape
    ky, kx = kernel.shape
    pad = xp.zeros_like(img, dtype=xp.float64)
    pad[:ky, :kx] = kernel
    pad = xp.roll(xp.roll(pad, -ky//2, axis=0), -kx//2, axis=1)
    Fimg = xp.fft.fft2(img)
    Fker = xp.fft.fft2(pad)
    conv = xp.fft.ifft2(Fimg * Fker)
    return xp.real(conv)

# ---------------- Acceleration strategy suggestion ----------------
_ACCEL_THRESHOLD = 1024  # if max dimension above this, enable fast detection by default

# Strategy implemented: downsample image by integer block-averaging to a target max dimension (~512)
# then compute FFT on small image, detect peaks, compute frequency vector (du,dv) from small FFT
# and map back to coordinates on original FFT (float coords -> round). This is fast and robust
# because peaks locations in frequency domain scale with image size.

# ----------------- GUI class -----------------
class FFTFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FFT Directional Filters POC')
        self.geometry('1200x800')

        # state
        self.img_cpu = None  # numpy grayscale image
        self.xp = np
        self.is_gpu = False
        self.cupy_available = False
        self.fft_log = None
        self.peaks_info = []
        self.kernels = []
        self.gradients = []
        self.timings = {}

        # UI layout
        self._build_controls()
        self._build_display()

    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(side='top', fill='x', padx=6, pady=6)

        btn_load = ttk.Button(frame, text='Load Image', command=self._on_load_image)
        btn_load.grid(row=0, column=0, padx=4)

        ttk.Label(frame, text='Device:').grid(row=0, column=1, padx=4)
        self.device_var = tk.StringVar(value='auto')
        cmb = ttk.Combobox(frame, textvariable=self.device_var, values=('auto','cpu','gpu'), width=6)
        cmb.grid(row=0, column=2, padx=4)

        self.fast_var = tk.BooleanVar(value=True)
        chk_fast = ttk.Checkbutton(frame, text='Fast detection (downsample for large images)', variable=self.fast_var)
        chk_fast.grid(row=0, column=3, padx=10)

        ttk.Label(frame, text='Adaptive k:').grid(row=0, column=4, padx=4)
        self.k_var = tk.DoubleVar(value=5.0)
        ent_k = ttk.Entry(frame, textvariable=self.k_var, width=6)
        ent_k.grid(row=0, column=5, padx=4)

        ttk.Label(frame, text='min_distance:').grid(row=0, column=6, padx=4)
        self.min_distance_var = tk.IntVar(value=10)
        spin_min = tk.Spinbox(frame, from_=1, to=200, textvariable=self.min_distance_var, width=6)
        spin_min.grid(row=0, column=7, padx=4)

        self.timer_var = tk.BooleanVar(value=True)
        chk_timer = ttk.Checkbutton(frame, text='Timer', variable=self.timer_var)
        chk_timer.grid(row=0, column=8, padx=8)

        btn_detect = ttk.Button(frame, text='Detect Peaks', command=self._on_detect)
        btn_detect.grid(row=0, column=9, padx=6)

        ttk.Label(frame, text='Detected filters:').grid(row=1, column=0, columnspan=1, sticky='w', pady=(6,0))
        # use Treeview to show position, sigma and angle clearly
        self.filter_tree = ttk.Treeview(frame, columns=('pos','sigma','angle'), show='headings', height=6)
        self.filter_tree.heading('pos', text='pos (r,c)')
        self.filter_tree.heading('sigma', text='sigma (px)')
        self.filter_tree.heading('angle', text='angle (deg)')
        self.filter_tree.column('pos', width=120, anchor='center')
        self.filter_tree.column('sigma', width=100, anchor='center')
        self.filter_tree.column('angle', width=100, anchor='center')
        self.filter_tree.grid(row=2, column=0, columnspan=6, sticky='w')
        self.filter_tree.bind('<<TreeviewSelect>>', self._on_select_filter)

        self.timing_label = ttk.Label(frame, text='')
        self.timing_label.grid(row=2, column=6, columnspan=4, sticky='w')

    def _build_display(self):
        # left: original image and FFT; right: kernel + gradient
        pan = ttk.Frame(self)
        pan.pack(fill='both', expand=True)

        left = ttk.Frame(pan)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(pan)
        right.pack(side='right', fill='both', expand=True)

        # figure for image
        self.fig_img, self.ax_img = plt.subplots(figsize=(4,4))
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left)
        self.canvas_img.get_tk_widget().pack(fill='both', expand=True)

        # figure for FFT
        self.fig_fft, self.ax_fft = plt.subplots(figsize=(4,4))
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=left)
        self.canvas_fft.get_tk_widget().pack(fill='both', expand=True)

        # kernel display
        self.fig_kernel, self.ax_kernel = plt.subplots(figsize=(4,4))
        self.canvas_kernel = FigureCanvasTkAgg(self.fig_kernel, master=right)
        self.canvas_kernel.get_tk_widget().pack(fill='both', expand=True)

        # gradient display
        self.fig_grad, self.ax_grad = plt.subplots(figsize=(4,4))
        self.canvas_grad = FigureCanvasTkAgg(self.fig_grad, master=right)
        self.canvas_grad.get_tk_widget().pack(fill='both', expand=True)

    # ---------- UI callbacks ----------
    def _on_load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images','*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif'), ('All','*.*')])
        if not path:
            return
        
        arr = plt.imread(path)
        self.img_cpu = ensure_gray(arr)
        self._draw_image()
        self._clear_results()

    def _clear_results(self):
        self.peaks_info = []
        self.kernels = []
        self.gradients = []
        # clear tree
        try:
            for item in self.filter_tree.get_children():
                self.filter_tree.delete(item)
        except Exception:
            pass
        self.ax_fft.clear(); self.canvas_fft.draw()
        self.ax_kernel.clear(); self.canvas_kernel.draw()
        self.ax_grad.clear(); self.canvas_grad.draw()

    def _draw_image(self):
        if self.img_cpu is None:
            return
        self.ax_img.clear()
        self.ax_img.imshow(self.img_cpu, cmap='gray', origin='upper')
        self.ax_img.set_title('Grayscale image')
        self.ax_img.axis('off')
        self.canvas_img.draw()

    def _on_detect(self):
        if self.img_cpu is None:
            messagebox.showerror('Error','Load an image first')
            return
        # set backend
        choice = self.device_var.get()
        try:
            self.xp, self.is_gpu, self.cupy_available = get_backend(choice)
        except Exception as e:
            messagebox.showerror('Device error', str(e))
            return
        # run detection (blocking) and time if requested
        timer = self.timer_var.get()
        fast = self.fast_var.get()
        adaptive_k = float(self.k_var.get())
        try:
            res = self._detect_pipeline(self.img_cpu, xp=self.xp, is_gpu=self.is_gpu, timer=timer, fast=fast, adaptive_k=adaptive_k)
        except Exception as e:
            messagebox.showerror('Detection error', str(e))
            return
        # fill treeview
        self.peaks_info = res['peaks_info']
        self.kernels = res['kernels']
        self.gradients = res['gradients']
        self.fft_log = res['fft_log']
        self.timings = res.get('timings', {})
        # clear tree
        for item in self.filter_tree.get_children():
            self.filter_tree.delete(item)
        for i,p in enumerate(self.peaks_info, start=1):
            r = p['r']; c = p['c']
            self.filter_tree.insert('', tk.END, values=(f'({r},{c})', f'{p["sigma"]:.2f}', f'{p["angle"]*180/math.pi:.1f}°'))
        self._draw_fft()  # draw FFT with markers
        self._update_timings()
        for i,p in enumerate(self.peaks_info, start=1):
            txt = f"{i}: pos=({p['r']},{p['c']}), σ={p['sigma']:.2f}px, θ={p['angle']*180/math.pi:.1f}°"
            self.filter_list.insert(tk.END, txt)
        self._draw_fft()  # draw FFT with markers
        self._update_timings()

    def _update_timings(self):
        if not self.timings:
            self.timing_label.config(text='')
            return
        txt = ' | '.join([f"{k}:{v:.3f}s" for k,v in self.timings.items()])
        self.timing_label.config(text=txt)

    def _draw_fft(self, highlight_idx=None):
        if self.fft_log is None:
            return
        self.ax_fft.clear()
        self.ax_fft.imshow(self.fft_log, origin='upper')
        self.ax_fft.set_title('FFT log magnitude (shifted)')
        self.ax_fft.axis('off')
        # overlay markers for each detected peak
        for i,p in enumerate(self.peaks_info, start=1):
            c = p['c']; r = p['r']
            self.ax_fft.scatter([c], [r], marker='o', edgecolors='red', facecolors='none', s=40, linewidths=1.2)
            if highlight_idx is not None and i-1 == highlight_idx:
                self.ax_fft.scatter([c], [r], marker='x', color='cyan', s=60)
        self.canvas_fft.draw()

    def _on_select_filter(self, evt):
        # handle selection from Treeview
        sel = self.filter_tree.selection()
        if not sel:
            return
        item = sel[0]
        idx = self.filter_tree.index(item)
        self._show_filter(idx)
        idx = sel[0]
        self._show_filter(idx)

    def _show_filter(self, idx):
        # show kernel and gradient and mark fft
        kernel = self.kernels[idx]
        grad = self.gradients[idx]
        self.ax_kernel.clear(); self.ax_kernel.imshow(kernel, origin='upper'); self.ax_kernel.set_title('Kernel (signed)'); self.ax_kernel.axis('off')
        self.canvas_kernel.draw()
        self.ax_grad.clear(); self.ax_grad.imshow(grad, cmap='viridis', origin='upper'); self.ax_grad.set_title('Gradient magnitude'); self.ax_grad.axis('off')
        self.canvas_grad.draw()
        # redraw FFT highlighting idx
        self._draw_fft(highlight_idx=idx)

    # ---------- core detection pipeline (single-threaded) ----------
    def _detect_pipeline(self, img_cpu, xp, is_gpu, timer, fast, adaptive_k):
        timings = {}
        def tic(k):
            if timer:
                sync_if_gpu(xp)
                timings[k + '_t0'] = time.perf_counter()
        def toc(k):
            if timer:
                sync_if_gpu(xp)
                t0 = timings.pop(k + '_t0', None)
                timings[k] = (time.perf_counter() - t0) if t0 is not None else None

        tic('total')
        img = img_cpu.astype(np.float64)

        # choose downsample factor if fast enabled and image large
        down_factor = 1
        maxdim = max(img.shape)
        if fast and maxdim > _ACCEL_THRESHOLD:
            # target approx max dimension 1024
            target = 1024
            down_factor = math.ceil(maxdim / target)
        # downsample for detection
        tic('downsample')
        if down_factor > 1:
            img_small = downsample_block_mean(img, down_factor)
        else:
            img_small = img
        toc('downsample')

        # move to backend if gpu
        tic('to_backend')
        if is_gpu:
            import cupy as cp
            img_b = cp.asarray(img_small)
        else:
            img_b = img_small
        toc('to_backend')

        # FFT on small image (fast) to get candidate frequency vectors
        tic('fft')
        Fsh_b, mag_b, mag_log_b = compute_fft_shift_mag(xp, img_b)
        # move magnitude to cpu for detection
        mag_b_cpu = to_cpu(xp, mag_b)
        toc('fft')

        # adaptive threshold and local maxima on small image (CPU)
        tic('threshold')
        thresh, med, rstd = adaptive_threshold_np(mag_b_cpu, k=adaptive_k)
        toc('threshold')

        tic('candidates')
        # default minimum distance for local maxima suppression on small image
        min_distance_local = int(self.min_distance_var.get())
        candidates_small = local_maxima_candidates_np(mag_b_cpu, threshold=thresh, min_distance=max(4, min_distance_local//down_factor))
        # exclude DC
        ny_s, nx_s = mag_b_cpu.shape
        cy_s, cx_s = ny_s//2, nx_s//2
        candidates_small = [c for c in candidates_small if (c[0]-cy_s)**2 + (c[1]-cx_s)**2 > (6**2)]
        toc('candidates')

        # pick top candidates and remove conjugates on small-scale
        tic('pick')
        cand_map = {(r,c):val for (r,c,val) in candidates_small}
        picked_small = []
        used = set()
        for (r,c,val) in sorted(candidates_small, key=lambda x:x[2], reverse=True):
            if (r,c) in used:
                continue
            r2,c2 = mirror_coord((r,c),(ny_s,nx_s))
            if (r2,c2) in cand_map and (r2,c2) not in used:
                val2 = cand_map[(r2,c2)]
                if val >= val2:
                    picked_small.append((r,c,val)); used.add((r,c)); used.add((r2,c2))
                else:
                    picked_small.append((r2,c2,val2)); used.add((r,c)); used.add((r2,c2))
            else:
                picked_small.append((r,c,val)); used.add((r,c))
            if len(picked_small) >= 20:
                break
        if not picked_small:
            # fallback: global top-n on small mag
            flat_idx = np.unravel_index(np.argsort(mag_b_cpu.ravel())[::-1], mag_b_cpu.shape)
            for i in range(len(flat_idx[0])):
                r, c = flat_idx[0][i], flat_idx[1][i]
                if (r-cy_s)**2 + (c-cx_s)**2 <= 6**2:
                    continue
                picked_small.append((r,c,mag_b_cpu[r,c]))
                if len(picked_small) >= 20:
                    break
        toc('pick')

        # Map small-image frequency vectors to full-size coordinates
        tic('map')
        ny, nx = img.shape
        cy, cx = ny//2, nx//2
        peaks_info = []
        for (r_s, c_s, val_s) in picked_small:
            du = (c_s - cx_s) / float(nx_s)
            dv = (r_s - cy_s) / float(ny_s)
            # map to original image's frequencies: position in full FFT
            # corresponding column in full FFT: c = cx + du * nx
            c_full = int(round(cx + du * nx))
            r_full = int(round(cy + dv * ny))
            # compute wavelength and sigma (in pixels of original image)
            f = math.hypot(du, dv)
            if f == 0:
                continue
            wavelength = 1.0 / f
            sigma = wavelength / (2.0 * math.pi)
            angle = math.atan2(dv, du)
            peaks_info.append({'r': r_full, 'c': c_full, 'val': float(val_s), 'wavelength': wavelength, 'sigma': sigma, 'angle': angle, 'du':du, 'dv':dv})
        toc('map')

        # merge similar sigmas (keep stronger peaks)
        tic('merge')
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
            if len(kept) >= 12:
                break
        peaks_info = kept
        toc('merge')

        # Build kernels (CPU arrays) and compute gradient maps on selected backend
        tic('kernels_conv')
        kernels = []
        grads = []
        # move full-res image to backend if needed
        if is_gpu:
            import cupy as cp
            img_bfull = cp.asarray(img)
        else:
            img_bfull = img
        for p in peaks_info:
            sigma = p['sigma']
            angle = p['angle']
            # build kernel on CPU (we visualize it and convolve on backend). Using numpy for kernel creation
            kernel_np = directional_gaussian_derivative_kernel(sigma, angle, truncate=3.0)
            kernels.append(kernel_np)
            # convolution on backend
            if is_gpu:
                import cupy as cp
                kernel_b = cp.asarray(kernel_np)
                conv = fft_convolve2d_backend(cp, img_bfull, kernel_b)
                grad = cp.abs(conv)
                grad_cpu = cp.asnumpy(grad)
            else:
                conv = fft_convolve2d_backend(np, img_bfull, kernel_np)
                grad_cpu = np.abs(conv)
            grads.append(grad_cpu)
        toc('kernels_conv')

        # compute full FFT log for visualization (CPU)
        tic('fft_full')
        xp_full = np if not is_gpu else __import__('cupy')
        Fsh_full, mag_full, mag_log_full = compute_fft_shift_mag(xp_full, img_bfull)
        mag_log_full_cpu = to_cpu(xp_full, mag_log_full)
        toc('fft_full')

        tic('total_end')
        toc('total')

        result = {
            'peaks_info': peaks_info,
            'kernels': kernels,
            'gradients': grads,
            'fft_log': mag_log_full_cpu,
            'threshold': None,
            'timings': {k:v for k,v in timings.items() if not k.endswith('_t0')}
        }
        return result

# ----------------- main -----------------
if __name__ == '__main__':
    app = FFTFilterGUI()
    app.mainloop()

# End of file
