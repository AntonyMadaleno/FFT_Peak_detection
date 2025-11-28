import math
import numpy as np

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