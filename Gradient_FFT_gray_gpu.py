# -*- coding: utf-8 -*-
"""
POC: directional gaussian-derivative filters from FFT peaks with optional GPU support (CuPy)
- Improved peak detection (adaptive threshold, remove conjugates, enforce sigma diversity)
- Directional derivative-of-Gaussian kernels built on the frequency orientation
- FFT-based convolution
- Optional GPU acceleration using CuPy with automatic fallback to NumPy
- Optional timing of blocks (timer=True) — reports times for major steps

Usage (from command line or import):
  - Run as script to execute a demo on a synthetic image:
      python poc_gpu_fft_filters.py
  - From Python, import improved_poc_gpu and call it:
      from poc_gpu_fft_filters import improved_poc_gpu
      result = improved_poc_gpu(image=my_gray_array, use_gpu=True, timer=True)

All comments in English. File encoded UTF-8.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, atan2, pi, cos, sin

# ----------------- Backend selection (NumPy / CuPy) -----------------

def get_backend(use_gpu=None):
    """
    Return (xp, is_gpu, cupy_available). xp behaves like numpy API (cupy if GPU).
    - use_gpu: None -> auto (use CuPy if available); True -> try CuPy (raise if not found);
             False -> force NumPy.
    """
    try:
        import cupy as cp
        cupy_available = True
    except Exception:
        cp = None
        cupy_available = False

    if use_gpu is None:
        use_gpu = cupy_available

    if use_gpu and not cupy_available:
        raise RuntimeError("CuPy requested but not available in this environment.")

    if use_gpu:
        xp = cp
        is_gpu = True
    else:
        xp = np
        is_gpu = False

    return xp, is_gpu, cupy_available


def to_cpu(xp, arr):
    """Return a NumPy array regardless of backend (copy from GPU if needed)."""
    if xp is np:
        return arr
    else:
        return xp.asnumpy(arr)


def sync_if_gpu(is_gpu):
    """Synchronize GPU to ensure accurate timing when using CuPy."""
    if is_gpu:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

# ----------------- low-level building blocks -----------------

def compute_fft_shift_mag(xp, img):
    """
    Compute shifted FFT and magnitude and log-magnitude
    img must be xp array on correct backend
    returns (Fshift, mag, mag_log) as xp arrays
    """
    F = xp.fft.fft2(img)
    Fsh = xp.fft.fftshift(F)
    mag = xp.abs(Fsh)
    # use log1p for visualization stability
    mag_log = xp.log1p(mag)
    return Fsh, mag, mag_log


def fft_convolve2d_backend(xp, img, kernel):
    """
    Convolve image with kernel using FFT on chosen backend.
    Both img and kernel are xp arrays. Kernel is padded and centered.
    Returns xp real array.
    """
    ny, nx = img.shape
    ky, kx = kernel.shape
    pad = xp.zeros_like(img, dtype=xp.float64)
    pad[:ky, :kx] = kernel
    # center kernel by rolling
    pad = xp.roll(xp.roll(pad, -ky//2, axis=0), -kx//2, axis=1)
    Fimg = xp.fft.fft2(img)
    Fker = xp.fft.fft2(pad)
    conv = xp.fft.ifft2(Fimg * Fker)
    return xp.real(conv)

# ----------------- peak detection utilities -----------------

def mirror_coord(coord, shape):
    r, c = coord
    ny, nx = shape
    cy, cx = ny//2, nx//2
    dr, dc = r - cy, c - cx
    r2 = cy - dr
    c2 = cx - dc
    return (r2, c2)


def adaptive_threshold_np(mag_np, k=4.0):
    """Compute adaptive threshold on a NumPy 2D magnitude array."""
    flat = mag_np.ravel()
    med = np.median(flat)
    mad = np.median(np.abs(flat - med))
    robust_std = 1.4826 * mad if mad > 0 else flat.std()
    thresh = med + k * robust_std
    return float(thresh), float(med), float(robust_std)


def local_maxima_candidates_np(mag_np, threshold, min_distance=8):
    """
    Simple local maxima detection on NumPy array. Non-maximum suppression by distance.
    Returns list of (r,c,val) sorted by val desc.
    """
    ny, nx = mag_np.shape
    candidates = []
    # scan interior (avoid borders)
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

# ----------------- kernel builders -----------------

def directional_gaussian_derivative_kernel_backend(xp, sigma, angle_rad, truncate=3.0):
    """
    Build directional derivative-of-Gaussian kernel on chosen backend.
    Derivative along x' axis rotated by angle_rad.
    Returns kernel as xp array (float64).
    """
    if sigma <= 0:
        sigma = 0.5
    radius = int(ceil(truncate * sigma))
    ys = xp.arange(-radius, radius+1)
    xs = xp.arange(-radius, radius+1)
    yy, xx = xp.meshgrid(ys, xs, indexing='ij')
    ct = cos(angle_rad); st = sin(angle_rad)
    # note: xx,yy are xp arrays only when xp is numpy; if xp is cupy, they are xp arrays as well
    xprime = ct * xx + st * yy
    yprime = -st * xx + ct * yy
    rsq = xprime**2 + yprime**2
    g = xp.exp(-rsq / (2.0 * sigma * sigma))
    dgdxp = - (xprime / (sigma * sigma)) * g
    s = float(xp.sum(xp.abs(dgdxp)))
    if s != 0.0:
        dgdxp = dgdxp / s
    return dgdxp.astype(xp.float64)

# ----------------- higher-level pipeline with timers -----------------

def improved_poc_gpu(image=None, use_gpu=None, top_k=6, min_distance=10, sigma_ratio_tol=1.25,
                     roi_halfsize=16, timer=False, adaptive_k=5.0, verbose=True):
    """
    Improved pipeline with optional GPU support and timing.

    Parameters
    - image: 2D numpy array (grayscale) or None to use synthetic image
    - use_gpu: None=auto, True=request CuPy, False=force NumPy
    - top_k: maximum candidate peaks
    - min_distance: spacing for non-maximum suppression
    - sigma_ratio_tol: merge tolerance for similar sigmas
    - roi_halfsize: ROI half-size (pixels) for FFT zoom display
    - timer: if True, measure execution time of major blocks
    - adaptive_k: multiplier for adaptive threshold
    - verbose: if True, show plots and print summaries

    Returns result dict with timings under 'timings' if timer=True.
    """
    # backend
    xp, is_gpu, cupy_available = get_backend(use_gpu)

    # timers
    timings = {}
    def tic(name):
        if timer:
            sync_if_gpu(is_gpu)
            timings[name + '_t0'] = time.perf_counter()
    def toc(name):
        if timer:
            sync_if_gpu(is_gpu)
            t0 = timings.pop(name + '_t0', None)
            if t0 is None:
                timings[name] = None
            else:
                timings[name] = time.perf_counter() - t0

    # 0) image creation / conversion
    tic('total')
    if image is None:
        # synthetic image on CPU (numpy), then copy to backend if needed
        img_cpu = _make_synthetic_image_np((256, 256))
    else:
        # accept numpy array, convert to float64 and grayscale if RGB
        img_cpu = _ensure_gray_cpu(image)

    if is_gpu:
        import cupy as cp
        tic('to_gpu')
        img = cp.asarray(img_cpu)
        toc('to_gpu')
    else:
        img = img_cpu

    # 1) FFT
    tic('fft')
    Fsh, mag, mag_log = compute_fft_shift_mag(xp, img)
    toc('fft')

    # 2) adaptive threshold (compute on CPU for robustness)
    tic('threshold')
    mag_cpu = to_cpu(xp, mag)
    thresh, med, rstd = adaptive_threshold_np(mag_cpu, k=adaptive_k)
    toc('threshold')

    # 3) local maxima candidates (CPU) and exclude DC region
    tic('candidates')
    candidates = local_maxima_candidates_np(mag_cpu, threshold=thresh, min_distance=max(4, min_distance//2))
    ny, nx = mag_cpu.shape
    cy, cx = ny // 2, nx // 2
    candidates = [c for c in candidates if (c[0]-cy)**2 + (c[1]-cx)**2 > (6**2)]
    toc('candidates')

    # 4) remove conjugates and pick top_k (CPU)
    tic('pick')
    cand_map = {(r, c): val for (r, c, val) in candidates}
    picked = []
    used = set()
    for (r, c, val) in sorted(candidates, key=lambda x: x[2], reverse=True):
        if (r, c) in used:
            continue
        r2, c2 = mirror_coord((r, c), (ny, nx))
        if (r2, c2) in cand_map and (r2, c2) not in used:
            val2 = cand_map[(r2, c2)]
            if val >= val2:
                picked.append((r, c, val)); used.add((r, c)); used.add((r2, c2))
            else:
                picked.append((r2, c2, val2)); used.add((r, c)); used.add((r2, c2))
        else:
            picked.append((r, c, val)); used.add((r, c))
        if len(picked) >= top_k:
            break
    # fallback to global top if none
    if not picked:
        flat_idx = np.unravel_index(np.argsort(mag_cpu.ravel())[::-1], mag_cpu.shape)
        for i in range(len(flat_idx[0])):
            r, c = flat_idx[0][i], flat_idx[1][i]
            if (r-cy)**2 + (c-cx)**2 <= 6**2:
                continue
            picked.append((r, c, mag_cpu[r, c]))
            if len(picked) >= top_k:
                break
    toc('pick')

    # 5) compute wavelengths / sigmas / angles, merge similar sigmas (CPU)
    tic('analyze_peaks')
    peaks_info = []
    for (r, c, val) in picked:
        du = (c - cx) / float(nx)
        dv = (r - cy) / float(ny)
        f = (du*du + dv*dv)**0.5
        wavelength = float(1.0 / f) if f != 0 else float('inf')
        sigma = wavelength / (2.0 * pi)
        angle = atan2(dv, du)
        peaks_info.append({'r': r, 'c': c, 'val': float(val), 'wavelength': wavelength, 'sigma': sigma, 'angle': angle, 'du': du, 'dv': dv})

    peaks_info = _merge_similar_sigmas_cpu(peaks_info, ratio_tol=sigma_ratio_tol)
    toc('analyze_peaks')

    # 6) build directional kernels (on chosen backend) and convolve
    gradients = []
    kernels_info = []
    tic('kernels_and_conv')
    for p in peaks_info:
        sigma = p['sigma']
        angle = p['angle']
        kernel = directional_gaussian_derivative_kernel_backend(xp, sigma, angle, truncate=3.0)
        kernels_info.append({'kernel': kernel, 'sigma': sigma, 'angle': angle})
        # convolution (on backend)
        conv = fft_convolve2d_backend(xp, img, kernel)
        # gradient magnitude (absolute directional derivative)
        grad = xp.abs(conv)
        gradients.append({'sigma': sigma, 'angle': angle, 'grad': grad})
    toc('kernels_and_conv')

    # 7) prepare results (convert to CPU for plotting/inspection)
    tic('to_cpu_results')
    result = {}
    result['image'] = img_cpu
    result['fft_log'] = to_cpu(xp, mag_log)
    result['picked'] = picked
    result['peaks_info'] = peaks_info
    result['kernels'] = []
    result['gradients'] = []
    for k in kernels_info:
        result['kernels'].append(to_cpu(xp, k['kernel']))
    for g in gradients:
        result['gradients'].append(to_cpu(xp, g['grad']))
    result['threshold'] = thresh
    result['median'] = med
    result['robust_std'] = rstd
    toc('to_cpu_results')

    tic('total_end')
    toc('total')
    toc('total_end')

    if timer:
        result['timings'] = timings

    # Visualization and prints if requested
    if verbose:
        _visualize_results(result, peaks_info, roi_halfsize)
        if timer:
            print('\nTimings (seconds):')
            for k, v in result.get('timings', {}).items():
                print(f'  {k}: {v:.6f}')

    return result

# ----------------- helper utilities (CPU-only helpers) -----------------

def _make_synthetic_image_np(shape=(256,256)):
    ny, nx = shape
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    img = 0.4 * np.sin(2*pi*(3*x/nx + 0.0*y/ny))
    img += 0.35 * np.sin(2*pi*(20*(cos(0.6)*x + sin(0.6)*y)/nx))
    img += 0.25 * np.sin(2*pi*(40*(cos(-0.3)*x + sin(-0.3)*y)/nx))
    gx = np.exp(-((x-nx*0.3)**2 + (y-ny*0.7)**2) / (2*(8.0**2)))
    gy = np.exp(-((x-nx*0.8)**2 + (y-ny*0.2)**2) / (2*(12.0**2)))
    img += 0.6*gx + 0.5*gy
    img = img - img.min(); img = img / img.max()
    return img


def _ensure_gray_cpu(image):
    arr = np.array(image, dtype=np.float64)
    if arr.ndim == 3:
        # simple mean conversion to grayscale
        arr = arr.mean(axis=2)
    # normalize to [0,1]
    arr = arr - arr.min()
    if arr.max() != 0:
        arr = arr / arr.max()
    return arr


def _merge_similar_sigmas_cpu(peaks_info, ratio_tol=1.25):
    if not peaks_info:
        return []
    peaks_info = sorted(peaks_info, key=lambda x: x['val'], reverse=True)
    kept = []
    for p in peaks_info:
        similar = False
        for q in kept:
            r1 = p['sigma'] / q['sigma'] if q['sigma'] > 0 else float('inf')
            r2 = q['sigma'] / p['sigma'] if p['sigma'] > 0 else float('inf')
            if r1 <= ratio_tol and r2 <= ratio_tol:
                similar = True
                break
        if not similar:
            kept.append(p)
    return kept

# ----------------- visualization (CPU arrays) -----------------

def _visualize_results(result, peaks_info, roi_halfsize=16):
    img = result['image']
    fft_log = result['fft_log']
    kernels = result['kernels']
    grads = result['gradients']

    plt.figure(figsize=(5,5)); plt.imshow(img, cmap='gray', origin='upper'); plt.title('Original image'); plt.axis('off'); plt.show()

    plt.figure(figsize=(5,5)); plt.imshow(fft_log, origin='upper'); plt.title('FFT magnitude (log1p) - shifted'); plt.axis('off'); plt.show()

    print(f"Adaptive threshold used: {result['threshold']:.3e} (median={result['median']:.3e}, robust_std~={result['robust_std']:.3e})")
    print('Detected peaks:')
    for i,p in enumerate(peaks_info, start=1):
        print(f"  {i}) pos=({p['r']},{p['c']}), mag={p['val']:.3e}, wl={p['wavelength']:.2f}px, sigma={p['sigma']:.3f}px, angle={p['angle']*180/pi:.1f}°")

    ny, nx = fft_log.shape
    cy, cx = ny//2, nx//2
    for i,p in enumerate(peaks_info, start=1):
        r,c = p['r'], p['c']; h = roi_halfsize
        r0 = max(0, r-h); r1 = min(ny, r+h+1); c0 = max(0, c-h); c1 = min(nx, c+h+1)
        roi = fft_log[r0:r1, c0:c1]
        plt.figure(figsize=(4,4)); plt.imshow(roi, origin='upper'); plt.scatter([c-c0],[r-r0], marker='x', color='cyan', s=50)
        plt.title(f'FFT ROI around peak {i} at ({r},{c})'); plt.axis('off'); plt.show()

        kernel = kernels[i-1]
        plt.figure(figsize=(4,4)); plt.imshow(kernel, origin='upper'); plt.colorbar(); plt.title(f'Kernel {i} (σ={p["sigma"]:.2f}px, θ={p["angle"]*180/pi:.1f}°)'); plt.axis('off'); plt.show()

    for i,g in enumerate(grads, start=1):
        plt.figure(figsize=(5,5)); plt.imshow(g, cmap='viridis', origin='upper'); plt.title(f'Gradient magnitude - directional filter {i}'); plt.axis('off'); plt.show()

# ----------------- CLI demo -----------------
if __name__ == '__main__':
    print('Running demo (GPU auto-detect). If you want to force CPU/GPU, call improved_poc_gpu from Python.')
    try:
        from tkinter import filedialog as fd
        img = plt.imread(fd.askopenfilename(title="Open an image"))   # or .jpg, .tif ...
        # convert to grayscale if RGB
        if img.ndim == 3:
            img = img.mean(axis=2)

        res = improved_poc_gpu(image=img, use_gpu=None, top_k=8, timer=True, verbose=True)
    except Exception as e:
        print('Demo failed:', e)
        print('Retrying with CPU fallback...')
        res = improved_poc_gpu(image=None, use_gpu=False, top_k=3, timer=True, verbose=True)

    print('\nDemo finished.')
