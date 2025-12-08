# -*- coding: utf-8 -*-
"""
Modified POC GUI: FFT-based filters with ellipse-based HSV gradient map
- In non-directional mode, combines 0° and 90° filters to create an HSV map
- Hue = gradient angle, Saturation = 100%, Value = ellipse volume
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
from matplotlib.colors import hsv_to_rgb

from FFT_utils import *
from Radial_utils import *

# ---------------- Backend selection (NumPy / CuPy) ----------------

def get_backend(use_gpu_choice=None):
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
    arr = np.array(image, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr - arr.min()
    if arr.max() != 0:
        arr = arr / arr.max()
    return arr


def downsample_block_mean(img, factor):
    if factor <= 1:
        return img
    ny, nx = img.shape
    ny2 = ny // factor
    nx2 = nx // factor
    img = img[:ny2*factor, :nx2*factor]
    img_ds = img.reshape(ny2, factor, nx2, factor).mean(axis=(1,3))
    return img_ds

# ---------------- Ellipse HSV mapping ----------------

def compute_ellipse_hsv_map(gx, gy, gamma = 2.2):
    """
    Combine Gx (0°) and Gy (90°) gradients into HSV map
    H = angle of gradient (from ellipse), S = 1.0, V = ellipse volume
    """

     # Try to detect CuPy without failing if it's not installed
    try:
        import cupy as cp
        _cupy_available = True
    except Exception:
        cp = None
        _cupy_available = False

    # If one (or both) of gx/gy are CuPy arrays, convert both to NumPy
    if _cupy_available and (isinstance(gx, cp.ndarray) or isinstance(gy, cp.ndarray)):
        gx = cp.asnumpy(gx) if isinstance(gx, cp.ndarray) else np.asarray(gx)
        gy = cp.asnumpy(gy) if isinstance(gy, cp.ndarray) else np.asarray(gy)
    else:
        # Ensure inputs are NumPy arrays (works if they are already np.ndarray)
        gx = np.asarray(gx)
        gy = np.asarray(gy)
        
    ny, nx = gx.shape
    hsv_map = np.zeros((ny, nx, 3), dtype=np.float32)
    
    # Compute angle and volume for each pixel
    angles = np.arctan2(gy, gx)  # angle in radians [-pi, pi]

    # Normalize to [0, 1] for hue
    hue = (angles + np.pi) / (2 * np.pi)
    
    # Norm
    # norm = np.sqrt(gx**2 * gy**2)
    norm = np.abs(gx) + np.abs(gy)
    
    # Normalize volume to [0, 1] for value
    norm_max = norm.max()

    if norm_max > 0:
        value = (norm / norm_max)**(1/gamma) # Value with gamma correction
    else:
        value = np.zeros_like(norm)
    
    # HSV: H = hue, S = 1.0, V = normalized volume
    hsv_map[:, :, 0] = hue
    hsv_map[:, :, 1] = 1.0  # Full saturation
    hsv_map[:, :, 2] = value
    
    # Convert HSV to RGB
    rgb_map = hsv_to_rgb(hsv_map)
    
    return rgb_map, angles, norm

# ---------------- ACCEL threshold ----------------
_ACCEL_THRESHOLD = 512

# ----------------- GUI class -----------------
class FFTFilterGUI(tk.Tk):
    def __init__(self):
        print("starting app\n")
        super().__init__()
        self.title('FFT Directional Filters POC - Ellipse HSV')
        self.geometry('1600x900')

        # state
        self.img_cpu = None
        self.xp = np
        self.is_gpu = False
        self.cupy_available = False
        self.fft_log = None
        self.peaks_info = []
        self.kernels = []
        self.gradients = []
        self.timings = {}
        self.hsv_map = None

        # UI
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
        chk_fast = ttk.Checkbutton(frame, text='Fast detection', variable=self.fast_var)
        chk_fast.grid(row=0, column=3, padx=10)

        ttk.Label(frame, text='Adaptive k:').grid(row=0, column=4, padx=4)
        self.k_var = tk.DoubleVar(value=5.0)
        ent_k = ttk.Entry(frame, textvariable=self.k_var, width=6)
        ent_k.grid(row=0, column=5, padx=4)

        ttk.Label(frame, text='min_distance:').grid(row=0, column=6, padx=4)
        self.min_distance_var = tk.IntVar(value=10)
        spin_min = tk.Spinbox(frame, from_=1, to=200, textvariable=self.min_distance_var, width=6)
        spin_min.grid(row=0, column=7, padx=4)

        ttk.Label(frame, text='max_gmm:').grid(row=0, column=8, padx=4)
        self.max_gmm = tk.IntVar(value=4)
        spin_min = tk.Spinbox(frame, from_=2, to=32, textvariable=self.max_gmm, width=6)
        spin_min.grid(row=0, column=9, padx=4)

        ttk.Label(frame, text='gamma:').grid(row=0, column=10, padx=4)
        self.gamma = tk.DoubleVar(value=1.0)
        spin_min = tk.Entry(frame, textvariable=self.gamma, width=6)
        spin_min.grid(row=0, column=11, padx=4)

        self.timer_var = tk.BooleanVar(value=True)
        chk_timer = ttk.Checkbutton(frame, text='Timer', variable=self.timer_var)
        chk_timer.grid(row=0, column=12, padx=8)

        self.directional_var = tk.BooleanVar(value=False)
        chk_dir = ttk.Checkbutton(frame, text='Directional filters', variable=self.directional_var)
        chk_dir.grid(row=0, column=13, padx=8)

        btn_detect = ttk.Button(frame, text='Detect Peaks', command=self._on_detect)
        btn_detect.grid(row=0, column=14, padx=6)

        ttk.Label(frame, text='Detected filters:').grid(row=1, column=0, columnspan=1, sticky='w', pady=(6,0))
        self.filter_tree = ttk.Treeview(frame, columns=('pos','sigma','angle','meanI'), show='headings', height=6)
        self.filter_tree.heading('pos', text='pos (r,c)')
        self.filter_tree.heading('sigma', text='sigma (px)')
        self.filter_tree.heading('angle', text='angle (deg)')
        self.filter_tree.heading('meanI', text='sum peak I')
        self.filter_tree.column('pos', width=120, anchor='center')
        self.filter_tree.column('sigma', width=100, anchor='center')
        self.filter_tree.column('angle', width=100, anchor='center')
        self.filter_tree.column('meanI', width=120, anchor='center')
        self.filter_tree.grid(row=2, column=0, columnspan=8, sticky='w')
        self.filter_tree.bind('<<TreeviewSelect>>', self._on_select_filter)

        self.timing_label = ttk.Label(frame, text='')
        self.timing_label.grid(row=2, column=8, columnspan=4, sticky='w')

    def _build_display(self):
        pan = ttk.Frame(self)
        pan.pack(fill='both', expand=True)

        left = ttk.Frame(pan)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(pan)
        right.pack(side='right', fill='both', expand=True)
        center = ttk.Frame(pan)
        center.pack(fill = 'both', expand=1)

        self.fig_img, self.ax_img = plt.subplots(figsize=(4,3.5))
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left)
        self.canvas_img.get_tk_widget().pack(fill='both', expand=True)

        self.fig_fft, self.ax_fft = plt.subplots(figsize=(4,3.5))
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=left)
        self.canvas_fft.get_tk_widget().pack(fill='both', expand=True)

        self.fig_kernel, self.ax_kernel = plt.subplots(figsize=(4,3.5))
        self.canvas_kernel = FigureCanvasTkAgg(self.fig_kernel, master=center)
        self.canvas_kernel.get_tk_widget().pack(fill='both', expand=True)

        self.fig_grad, self.ax_grad = plt.subplots(figsize=(4,3.5))
        self.canvas_grad = FigureCanvasTkAgg(self.fig_grad, master=center)
        self.canvas_grad.get_tk_widget().pack(fill='both', expand=True)

        self.fig_profile_fft, self.ax_profile_fft = plt.subplots(figsize=(4,6))
        self.canvas_profile_fft = FigureCanvasTkAgg(self.fig_profile_fft, master=right)
        self.canvas_profile_fft.get_tk_widget().pack(fill='both', expand=True)

        self.fig_profile_grad, self.ax_profile_grad = plt.subplots(figsize=(4,6))
        self.canvas_profile_grad = FigureCanvasTkAgg(self.fig_profile_grad, master=right)
        self.canvas_profile_grad.get_tk_widget().pack(fill='both', expand=True)

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
        self.hsv_map = None
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
        self.canvas_img.draw()

    def _on_detect(self):
        if self.img_cpu is None:
            messagebox.showerror('Error','Load an image first')
            return
        choice = self.device_var.get()
        try:
            self.xp, self.is_gpu, self.cupy_available = get_backend(choice)
        except Exception as e:
            messagebox.showerror('Device error', str(e))
            return
        timer = self.timer_var.get()
        fast = self.fast_var.get()
        adaptive_k = float(self.k_var.get())

        res = self._detect_pipeline(self.img_cpu, xp=self.xp, is_gpu=self.is_gpu, timer=timer, fast=fast, adaptive_k=adaptive_k, directional=self.directional_var.get())
        
        self.peaks_info = res['peaks_info']
        self.kernels = res['kernels']
        self.gradients = res['gradients']
        self.fft_log = res['fft_log']
        self.timings = res.get('timings', {})
        self.hsv_map = res.get('hsv_map', None)
        self.hsv_maps = res.get('hsv_maps', None)
        self.radii, self.profile = res.get('fft_profile', None)
        self.gmm_info = res.get('GMM info', None)
        
        # populate tree
        for item in self.filter_tree.get_children():
            self.filter_tree.delete(item)
        for i,p in enumerate(self.peaks_info, start=1):
            pos = p.get('pos', ('-', '-'))
            sigma = p.get('sigma', 0.0)
            angle_deg = p.get('angle', 0.0) * 180.0 / math.pi
            meanI = p.get('meanI', 0.0)
            self.filter_tree.insert('', tk.END, values=(f'({pos[0]},{pos[1]})', f'{sigma:.2f}', f'{angle_deg:.1f}°', f'{meanI:.3f}'))
        self._draw_fft()
        self._update_timings()
        
        # Display HSV map if available
        if self.hsv_map is not None:
            self.ax_grad.clear()
            self.ax_grad.imshow(self.hsv_map, origin='upper')
            self.ax_grad.set_title('HSV Gradient Map (H=angle, V=volume)')
            self.ax_grad.axis('off')
            self.canvas_grad.draw()

    def _update_timings(self):
        if not self.timings:
            self.timing_label.config(text='')
            return
        txt = ' | '.join([f"{k}:{v:.3f}s" for k,v in self.timings.items()])
        self.timing_label.config(text=txt)

    def _draw_fft(self, highlight_idx=None):
        if self.fft_log is None:
            return
        
        width, height =  self.fft_log.shape
        self.ax_fft.clear()
        self.ax_fft.imshow(self.fft_log, origin='upper', cmap='magma')
        self.ax_fft.set_title('FFT log magnitude')
        self.ax_fft.axis('off')

        self.ax_profile_fft.clear()
        self.ax_profile_fft.plot(np.log2(np.min([width, height]) / self.radii[1:]), self.profile[1:], lw= 2,  alpha= 0.5)
        self.ax_profile_fft.set_title('FFT Radial profile')

        x = np.arange(start = 1, stop = len(self.gmm_info['modeled']))
        self.ax_profile_fft.plot(np.log2(np.min([width, height]) / x), self.gmm_info['modeled'][1:], label=f'GMM mixture', lw= 2, alpha = 0.5)

        for num, c in enumerate(self.gmm_info['components']):
            self.ax_profile_fft.plot(np.log2(np.min([width, height]) / x), c['pdf'][1:], lw= 1, linestyle='--', alpha= 0.5, label = f"composante {num}")
        
        # marques des centres sélectionnés
        for (r, v) in self.gmm_info['selected']:
            self.ax_profile_fft.axvline(np.log2(np.min([width, height]) / r), color='k', linestyle=':', alpha=0.6)
            self.ax_profile_fft.scatter([np.log2(np.min([width, height]) / r)], [v], c='k')

        for i,p in enumerate(self.peaks_info, start=1):
            c = p.get('c', None)
            r = p.get('r', None)
            if c is None or r is None:
                continue
            self.ax_fft.scatter([c], [r], marker='o', edgecolors='red', facecolors='none', s=20, linewidths=2)
            radius = int(np.sqrt((c - height / 2)**2 + (r - width / 2)**2))
            self.ax_profile_fft.scatter(np.log2(np.min([width, height]) / float(radius)), self.profile[radius], marker='o', edgecolors='red', facecolors='none', s=20, linewidths=2)
            if highlight_idx is not None and i-1 == highlight_idx:
                self.ax_fft.scatter([c], [r], marker='x', color='cyan', s=50)
                radius = int(np.sqrt((c - height / 2)**2 + (r - width / 2)**2))
                self.ax_profile_fft.scatter(np.log2(np.min([width, height]) / float(radius)), self.profile[radius], color = "cyan", marker = 'x', s=50)

                self.ax_profile_grad.clear()
                angles, magnitudes = direction_profile(self.fft_mag, distance= radius)
                self.ax_profile_grad.plot(angles[0:int(len(angles)/2)], magnitudes[0:int(len(angles)/2)], color="#33ff33", lw= 2,  alpha= 0.5)
                self.ax_profile_grad.set_title(f"FFT Angle Profile {radius}px")

                self.canvas_profile_grad.draw()

        self.canvas_profile_fft.draw()
        self.canvas_fft.draw()

    def _on_select_filter(self, evt):
        sel = self.filter_tree.selection()
        if not sel:
            return
        item = sel[0]
        idx = self.filter_tree.index(item)
        self._show_filter(idx)

    def _show_filter(self, idx):
        if idx >= len(self.kernels):
            return
        kernel = self.kernels[idx]
        grad = self.gradients[idx]
        
        self.ax_kernel.clear()
        ky, kx = kernel.shape
        extent = (0, kx, 0, ky)
        self.ax_kernel.imshow(kernel, origin='upper', extent=extent)
        self.ax_kernel.set_title(f'Kernel {idx} – {ky}x{kx}')
        self.ax_kernel.set_xlabel('x (px)')
        self.ax_kernel.set_ylabel('y (px)')
        self.canvas_kernel.draw()

        # Show individual gradient if not in HSV mode
        if not self.directional_var:
            self.ax_grad.clear()
            self.ax_grad.imshow(grad, cmap='viridis', origin='upper')
            self.ax_grad.set_title(f'Gradient {idx}')
            self.ax_grad.axis('off')
            self.canvas_grad.draw()
        else:
            idx_hsv = int(np.floor(idx / 2))
            self.ax_grad.clear()
            self.ax_grad.imshow(self.hsv_maps[idx_hsv][0])
            self.ax_grad.set_title(f'Gradient {idx_hsv}')
            self.ax_grad.axis('off')
            self.canvas_grad.draw()

        self.ax_profile_fft.clear()
        self.ax_profile_fft.plot(self.radii, self.profile, lw = 2, alpha = 0.5, color = '#ff6600')
        self.ax_profile_fft.set_title(f'Profile FFT')
        self.canvas_profile_fft.draw()


        self._draw_fft(highlight_idx=idx)

    def _detect_pipeline(self, img_cpu, xp, is_gpu, timer, fast, adaptive_k, directional=True):
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

        down_factor = 1
        maxdim = max(img.shape)
        if fast and maxdim > _ACCEL_THRESHOLD:
            target = 1024
            down_factor = math.ceil(maxdim / target)
        
        tic('downsample')
        if down_factor > 1:
            img_small = downsample_block_mean(img, down_factor)
        else:
            img_small = img
        toc('downsample')

        tic('to_backend')
        if is_gpu:
            import cupy as cp
            img_b = cp.asarray(img_small)
        else:
            img_b = img_small
        toc('to_backend')

        tic('fft')
        Fsh_b, mag_b, mag_log_b = compute_fft_shift_mag(xp, img_b)
        mag_b_cpu = to_cpu(xp, mag_b)
        toc('fft')

        peaks_info = []
        hsv_map = None

        if directional:
            # Original directional path
            tic('threshold')
            thresh, med, rstd = adaptive_threshold_np(mag_b_cpu, k=adaptive_k)
            toc('threshold')

            tic('candidates')
            min_distance_local = int(self.min_distance_var.get())
            candidates_small = local_maxima_candidates_np(mag_b_cpu, threshold=thresh, min_distance=max(4, min_distance_local//down_factor))
            ny_s, nx_s = mag_b_cpu.shape
            cy_s, cx_s = ny_s//2, nx_s//2
            candidates_small = [c for c in candidates_small if (c[0]-cy_s)**2 + (c[1]-cx_s)**2 > (6**2)]
            toc('candidates')

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
                flat_idx = np.unravel_index(np.argsort(mag_b_cpu.ravel())[::-1], mag_b_cpu.shape)
                for i in range(len(flat_idx[0])):
                    r, c = flat_idx[0][i], flat_idx[1][i]
                    if (r-cy_s)**2 + (c-cx_s)**2 <= 6**2:
                        continue
                    picked_small.append((r,c,mag_b_cpu[r,c]))
                    if len(picked_small) >= 20:
                        break
            toc('pick')

            tic('map')
            ny, nx = img.shape
            cy, cx = ny//2, nx//2
            for (r_s, c_s, val_s) in picked_small:
                du = (c_s - cx_s) / float(nx_s)
                dv = (r_s - cy_s) / float(ny_s)
                c_full = int(round(cx + du * nx))
                r_full = int(round(cy + dv * ny))
                f = math.hypot(du, dv)
                if f == 0:
                    continue
                wavelength = 1.0 / f
                sigma = wavelength / (2.0 * math.pi)
                angle = math.atan2(dv, du)
                peaks_info.append({'r': r_full, 'c': c_full, 'val': float(val_s), 'wavelength': wavelength, 'sigma': sigma, 'angle': angle, 'du':du, 'dv':dv})
            toc('map')

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

        else:
            # Non-directional: radial profile approach
            tic('fft_full_radial')
            xp_full = np if not is_gpu else __import__('cupy')
            if is_gpu:
                import cupy as cp
                img_bfull = cp.asarray(img)
            else:
                img_bfull = img
            Fsh_full, mag_full, mag_log_full = compute_fft_shift_mag(xp_full, img_bfull)
            mag_full_cpu = to_cpu(xp_full, mag_full)
            try:
                self.fft_mag = mag_full_cpu
            except Exception:
                self.fft_mag = None
            toc('fft_full_radial')

            tic('radial_profile')
            ny, nx = mag_full_cpu.shape
            cy, cx = ny//2, nx//2
            max_r = min(cy, cx, ny-cy-1, nx-cx-1)
            radii, profile, counts = radial_distance_profile(mag_full_cpu, center=(cy,cx), max_radius=max_r)

            try:
                adaptive_k = adaptive_k  # deja définie
            except NameError:
                adaptive_k = 1.0

            # seuil ancien (facultatif, tu peux garder ou retirer) :
            med = np.median(profile)
            std = profile.std()
            thresh = med + adaptive_k * std

            min_r = max(1, 2)
            # tenter GMM
            min_dist_r = max(1, self.min_distance_var.get() // max(1, down_factor))
            sel_gmm, comps, modeled, k_best, f_vals, Ks = select_peaks_with_gmm_and_components(profile[min_r:], max_peaks= self.max_gmm.get(), min_distance=min_dist_r)
            # note : on a passé profile[min_r:] ; il faut ajuster les indices retournés
            if sel_gmm is None:
                # fallback heuristique original si GMM a échoué
                candidates = [(r, float(profile[r])) for r in range(min_r, len(profile))]
                candidates.sort(key=lambda x: x[1], reverse=True)
                sel = []
                for (r, val) in candidates:
                    if val < thresh:
                        continue
                    too_close = False
                    for (sr, sval) in sel:
                        if abs(sr - r) <= min_dist_r:
                            too_close = True
                            break
                    if not too_close:
                        sel.append((r, val))
                    if len(sel) >= 4:
                        break
            else:
                # réajuster indices (car on a passé profile[min_r:])
                sel = [(r + min_r, v) for (r, v) in sel_gmm]
                # si aucun trouvé ou valeurs en dessous du threshold -> retomber sur heuristique (optionnel)
                if not sel:
                    # fallback heuristique conservateur
                    candidates = [(r, float(profile[r])) for r in range(min_r, len(profile))]
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    sel = []
                    for (r, val) in candidates:
                        too_close = False
                        for (sr, sval) in sel:
                            if abs(sr - r) <= min_dist_r:
                                too_close = True
                                break
                        if not too_close:
                            sel.append((r, val))
                        if len(sel) >= self.max_gmm:
                            break


            toc('radial_profile')

            tic('map_radial')
            diag = math.hypot(nx, ny)
            for (r_pix, val) in sel:
                if r_pix == 0:
                    continue
                f = float(r_pix) / diag
                if f == 0:
                    continue
                wavelength = 1.0 / f
                sigma = wavelength / (2.0 * math.pi)
                meanI = float(profile[r_pix])
                # Generate two filters: 0° and 90°
                c0 = int(min(nx-1, cx + r_pix))
                r0 = int(cy)
                peaks_info.append({'r': r0, 'c': c0, 'val': float(val), 'wavelength': wavelength, 'sigma': sigma,
                                   'angle': 0.0, 'meanI': meanI, 'pos': (r0, c0)})
                c1 = int(cx)
                r1 = int(max(0, cy - r_pix))
                peaks_info.append({'r': r1, 'c': c1, 'val': float(val), 'wavelength': wavelength, 'sigma': sigma,
                                   'angle': math.pi/2.0, 'meanI': meanI, 'pos': (r1, c1)})
            toc('map_radial')

            tic('fft_full_log')
            mag_log_full_cpu = np.log1p(mag_full_cpu)
            toc('fft_full_log')
            self.fft_log = mag_log_full_cpu

        # Build kernels and convolutions
        tic('kernels_conv')
        kernels = []
        grads = []
        if is_gpu:
            import cupy as cp
            img_bfull = cp.asarray(img)
        else:
            img_bfull = img

        # For directional mode, compute full FFT magnitude for intensity
        if directional:
            tic('fft_full_for_int')
            xp_full = np if not is_gpu else __import__('cupy')
            Fsh_full, mag_full, mag_log_full = compute_fft_shift_mag(xp_full, img_bfull)
            mag_full_cpu = to_cpu(xp_full, mag_full)
            toc('fft_full_for_int')
            try:
                self.fft_mag = mag_full_cpu
            except Exception:
                self.fft_mag = None
            for p in peaks_info:
                r = int(p.get('r', 0)); c = int(p.get('c', 0))
                nyf, nxf = mag_full_cpu.shape
                r0 = max(0, r-1); r1 = min(nyf-1, r+1)
                c0 = max(0, c-1); c1 = min(nxf-1, c+1)
                window = mag_full_cpu[r0:r1+1, c0:c1+1]
                p['meanI'] = float(window.mean()) if window.size>0 else float(p.get('val',0.0))
                p['pos'] = (r, c)

        # Generate kernels and convolve
        for p in peaks_info:
            sigma = p['sigma'] if 'sigma' in p else p.get('sigma', 1.0)
            angle = p.get('angle', 0.0)
            kernel_np = directional_gaussian_derivative_kernel(sigma, angle, truncate=3.0)
            kernels.append(kernel_np)
            if is_gpu:
                import cupy as cp
                kernel_b = cp.asarray(kernel_np)
                conv = fft_convolve2d_backend(cp, img_bfull, kernel_b)
                # grad = cp.abs(conv)
                grad_cpu = cp.asnumpy(conv)
            else:
                conv = fft_convolve2d_backend(np, img_bfull, kernel_np)
                # grad_cpu = np.abs(conv)
            grads.append(conv)
        toc('kernels_conv')

        # Compute HSV map for non-directional mode
        if not directional and len(grads) >= 2:
            tic('hsv_map')
            # Group gradients by sigma (pairs of 0° and 90°)
            sigma_groups = {}
            for i, p in enumerate(peaks_info):
                sigma = p['sigma']
                angle = p['angle']
                if sigma not in sigma_groups:
                    sigma_groups[sigma] = {}
                sigma_groups[sigma][angle] = grads[i]
            
            # For each sigma, combine 0° and 90° gradients
            hsv_maps = []
            for sigma, angle_dict in sigma_groups.items():
                if 0.0 in angle_dict and math.pi/2.0 in angle_dict:
                    gx = angle_dict[0.0]
                    gy = angle_dict[math.pi/2.0]
                    rgb_map, angles, norm = compute_ellipse_hsv_map(gx, gy, gamma = self.gamma.get())
                    hsv_maps.append((rgb_map, norm.max()))
            
            # Combine all HSV maps by taking the one with max volume at each pixel
            if hsv_maps:
                combined_rgb = np.zeros_like(hsv_maps[0][0])
                max_norm = np.zeros(combined_rgb.shape[:2])
                for rgb_map, _ in hsv_maps:
                    # Extract volume from RGB (it's in the V channel originally)
                    # We need to recompute or track it - let's use a simpler approach
                    # and just average all maps weighted by their max intensity
                    combined_rgb += rgb_map
                combined_rgb /= len(hsv_maps)
                hsv_map = combined_rgb
            toc('hsv_map')

        # Compute full FFT log if not already done
        if getattr(self, 'fft_log', None) is None:
            tic('fft_full')
            xp_full = np if not is_gpu else __import__('cupy')
            Fsh_full, mag_full, mag_log_full = compute_fft_shift_mag(xp_full, img_bfull)
            mag_log_full_cpu = to_cpu(xp_full, mag_log_full)
            toc('fft_full')
            self.fft_log = mag_log_full_cpu

        toc('total')

        result = {
            'peaks_info': peaks_info,
            'kernels': kernels,
            'gradients': grads,
            'fft_log': self.fft_log,
            'hsv_map': hsv_map,
            'hsv_maps': hsv_maps,
            'fft_profile': (radii, profile),
            'GMM info': {
                'selected': sel,
                'components': comps, 
                'modeled': modeled
            },
            'timings': {k:v for k,v in timings.items() if not k.endswith('_t0')}
        }
        return result

 # ------------------- run -----------------------------------------------------
if __name__ == '__main__':
    app = FFTFilterGUI()
    app.mainloop()