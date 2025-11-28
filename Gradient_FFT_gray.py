# Retry execution (same improved POC code). Some execution environments reset between attempts;
# re-run in one cell.
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, atan2, pi, cos, sin

def make_synthetic_image(shape=(256,256)):
    ny, nx = shape
    y = np.arange(ny)[:,None]; x = np.arange(nx)[None,:]
    img = 0.4 * np.sin(2*np.pi*(3*x/nx + 0.0*y/ny))
    img += 0.35 * np.sin(2*np.pi*(20*(np.cos(0.6)*x + np.sin(0.6)*y)/nx))
    img += 0.25 * np.sin(2*np.pi*(40*(np.cos(-0.3)*x + np.sin(-0.3)*y)/nx))
    gx = np.exp(-((x-nx*0.3)**2+(y-ny*0.7)**2)/(2*(8.0**2)))
    gy = np.exp(-((x-nx*0.8)**2+(y-ny*0.2)**2)/(2*(12.0**2)))
    img += 0.6*gx + 0.5*gy
    img = img - img.min(); img = img / img.max()
    return img

def compute_fft_shift_mag(img):
    F = np.fft.fft2(img); Fsh = np.fft.fftshift(F)
    mag = np.abs(Fsh); mag_log = np.log1p(mag)
    return Fsh, mag, mag_log

def mirror_coord(coord, shape):
    r, c = coord; ny, nx = shape; cy, cx = ny//2, nx//2
    dr, dc = r - cy, c - cx
    r2 = cy - dr; c2 = cx - dc
    return (r2, c2)

def local_maxima_candidates(mag, threshold, min_distance=8):
    ny, nx = mag.shape
    candidates = []
    for r in range(1, ny-1):
        row = mag[r]
        if row.max() < threshold:
            continue
        for c in range(1, nx-1):
            val = mag[r,c]
            if val < threshold:
                continue
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
        if not too_close:
            kept.append((r,c,val))
    return kept

def adaptive_threshold(mag, k=4.0):
    flat = mag.ravel()
    med = np.median(flat); mad = np.median(np.abs(flat-med))
    robust_std = 1.4826 * mad if mad > 0 else np.std(flat)
    thresh = med + k * robust_std
    return thresh, med, robust_std

def peak_to_frequency_vector(peak, shape):
    ny, nx = shape; r,c = peak; cy, cx = ny//2, nx//2
    dv = (r - cy) / ny; du = (c - cx) / nx
    return du, dv

def freq_vector_to_wavelength(du, dv):
    f = np.sqrt(du*du + dv*dv)
    if f == 0: return np.inf
    return 1.0 / f

def merge_similar_sigmas(peaks_info, ratio_tol=1.2):
    if not peaks_info: return []
    peaks_info = sorted(peaks_info, key=lambda x: x['val'], reverse=True)
    kept = []
    for p in peaks_info:
        similar = False
        for q in kept:
            r1 = p['sigma'] / q['sigma'] if q['sigma']>0 else np.inf
            r2 = q['sigma'] / p['sigma'] if p['sigma']>0 else np.inf
            if r1 <= ratio_tol and r2 <= ratio_tol:
                similar = True; break
        if not similar: kept.append(p)
    return kept

def directional_gaussian_derivative_kernel(sigma, angle_rad, truncate=3.0):
    if sigma <= 0: sigma = 0.5
    radius = int(ceil(truncate * sigma))
    ys = np.arange(-radius, radius+1); xs = np.arange(-radius, radius+1)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    ct = cos(angle_rad); st = sin(angle_rad)
    xprime = ct * xx + st * yy
    yprime = -st * xx + ct * yy
    rsq = xprime**2 + yprime**2
    g = np.exp(-rsq/(2*sigma*sigma))
    dgdxp = - (xprime / (sigma*sigma)) * g
    s = np.sum(np.abs(dgdxp))
    if s != 0: dgdxp = dgdxp / s
    return dgdxp

def fft_convolve2d(img, kernel):
    ny, nx = img.shape; ky, kx = kernel.shape
    pad = np.zeros_like(img, dtype=np.float64)
    pad[:ky, :kx] = kernel
    pad = np.roll(np.roll(pad, -ky//2, axis=0), -kx//2, axis=1)
    Fimg = np.fft.fft2(img); Fker = np.fft.fft2(pad)
    conv = np.fft.ifft2(Fimg * Fker)
    return np.real(conv)

def improved_poc(image=None, top_k=6, min_distance=10, sigma_ratio_tol=1.25, roi_halfsize=16, show=True):
    if image is None: image = make_synthetic_image((256,256))
    ny, nx = image.shape
    Fsh, mag, mag_log = compute_fft_shift_mag(image)
    thresh, med, rstd = adaptive_threshold(mag, k=5.0)
    candidates = local_maxima_candidates(mag, threshold=thresh, min_distance=4)
    cy, cx = ny//2, nx//2
    candidates = [c for c in candidates if (c[0]-cy)**2 + (c[1]-cx)**2 > (6**2)]
    cand_map = {(r,c):val for (r,c,val) in candidates}
    picked = []; used = set()
    for (r,c,val) in sorted(candidates, key=lambda x: x[2], reverse=True):
        if (r,c) in used: continue
        r2, c2 = mirror_coord((r,c), (ny,nx))
        if (r2, c2) in cand_map and (r2,c2) not in used:
            val2 = cand_map[(r2,c2)]
            if val >= val2:
                picked.append((r,c,val)); used.add((r,c)); used.add((r2,c2))
            else:
                picked.append((r2,c2,val2)); used.add((r,c)); used.add((r2,c2))
        else:
            picked.append((r,c,val)); used.add((r,c))
        if len(picked) >= top_k: break
    if not picked:
        flat_idx = np.unravel_index(np.argsort(mag.ravel())[::-1], mag.shape)
        for i in range(len(flat_idx[0])):
            r, c = flat_idx[0][i], flat_idx[1][i]
            if (r-cy)**2 + (c-cx)**2 <= 6**2: continue
            picked.append((r,c,mag[r,c]))
            if len(picked) >= top_k: break
    peaks_info = []
    for (r,c,val) in picked:
        du, dv = peak_to_frequency_vector((r,c),(ny,nx))
        wavelength = freq_vector_to_wavelength(du, dv)
        sigma = wavelength / (2.0 * np.pi)
        angle = atan2(dv, du)
        peaks_info.append({'r':r,'c':c,'val':val,'wavelength':wavelength,'sigma':sigma,'angle':angle, 'du':du,'dv':dv})
    peaks_info = merge_similar_sigmas(peaks_info, ratio_tol=sigma_ratio_tol)
    gradients = []; kernels = []
    for p in peaks_info:
        sigma = p['sigma']; angle = p['angle']
        kernel = directional_gaussian_derivative_kernel(sigma, angle, truncate=3.0)
        kernels.append({'kernel': kernel, 'sigma': sigma, 'angle': angle})
        gx = fft_convolve2d(image, kernel)
        grad_mag = np.abs(gx)
        gradients.append({'sigma':sigma, 'angle':angle, 'kernel':kernel, 'grad':grad_mag})
    result = {
        'image': image, 'fft_shift': Fsh, 'fft_mag': mag, 'fft_log': mag_log,
        'picked_peaks': picked, 'peaks_info': peaks_info, 'kernels': kernels,
        'gradients': gradients, 'threshold': thresh, 'median': med, 'robust_std': rstd
    }
    if show:
        plt.figure(figsize=(5,5)); plt.imshow(image, cmap='gray', origin='upper')
        plt.title('Original grayscale image'); plt.axis('off'); plt.show()

        plt.figure(figsize=(5,5)); plt.imshow(mag_log, origin='upper')
        plt.title('FFT magnitude (log1p) - shifted (full view)'); plt.axis('off'); plt.show()

        print(f"Adaptive threshold: {thresh:.3e} (median={med:.3e}, robust_std≈{rstd:.3e})")
        print("Picked peaks (after conjugate removal and merging):")
        for i,p in enumerate(peaks_info, start=1):
            print(f"  {i}) pos=({p['r']},{p['c']}), mag={p['val']:.3e}, wl={p['wavelength']:.2f}px, sigma={p['sigma']:.3f}px, angle={p['angle']*180/pi:.1f}°")

        for i,p in enumerate(peaks_info, start=1):
            r,c = p['r'], p['c']; h = roi_halfsize
            r0 = max(0, r-h); r1 = min(ny, r+h+1); c0 = max(0, c-h); c1 = min(nx, c+h+1)
            roi = mag_log[r0:r1, c0:c1]
            plt.figure(figsize=(4,4)); plt.imshow(roi, origin='upper')
            plt.title(f'FFT ROI around peak {i} at ({r},{c}) (log mag)')
            plt.scatter([c - c0], [r - r0], marker='x', color='cyan', s=50)
            plt.axis('off'); plt.show()

            kernel = directional_gaussian_derivative_kernel(p['sigma'], p['angle'], truncate=3.0)
            plt.figure(figsize=(4,4)); plt.imshow(kernel, origin='upper')
            plt.title(f'Kernel {i} (σ={p["sigma"]:.2f}px, θ={p["angle"]*180/pi:.1f}°)')
            plt.colorbar(); plt.axis('off'); plt.show()

        if gradients:
            for i,g in enumerate(gradients, start=1):
                plt.figure(figsize=(5,5)); plt.imshow(g['grad'], cmap='viridis', origin='upper')
                plt.title(f'Gradient magnitude - directional filter {i}\nσ={g["sigma"]:.2f}px, θ={g["angle"]*180/pi:.1f}°')
                plt.axis('off'); plt.show()
        else:
            print("No gradients to display (no peaks detected).")
    return result

from tkinter import filedialog as fd
img = plt.imread(fd.askopenfilename(title="Open an image"))   # or .jpg, .tif ...
# convert to grayscale if RGB
if img.ndim == 3:
    img = img.mean(axis=2)

# call the pipeline (assumes run_poc function is defined in the script)
result = improved_poc(image=img, top_k=3, show=True)

print("\nConcise summary after improvements:")
for i,p in enumerate(result['peaks_info'], start=1):
    print(f"  Filter {i}: pos=({p['r']},{p['c']}), mag={p['val']:.3e}, wl={p['wavelength']:.2f}px, sigma={p['sigma']:.3f}px, angle={p['angle']*180/pi:.1f}°")