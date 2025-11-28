# Python code to load two same-size images, swap their Fourier phases, and display results.
# It tries to load 'image1.png' and 'image2.png' from the working directory.
# If they are not present, it falls back to skimage sample images if available,
# otherwise it generates synthetic images.
# The code handles grayscale and RGB images (per-channel FFT for RGB).
# It displays each image in its own matplotlib figure (no subplots) as requested.

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_or_fallback(path1='image1.png', path2='image2.png'):
    # Try to open files
    if os.path.exists(path1) and os.path.exists(path2):
        im1 = Image.open(path1).convert('RGB')
        im2 = Image.open(path2).convert('RGB')
        return np.array(im1), np.array(im2), path1, path2
    
    # Try skimage sample images
    try:
        from skimage import data, color
        img1 = data.astronaut()  # RGB
        img2 = data.camera()     # grayscale
        if img2.ndim == 2:
            img2 = color.gray2rgb(img2)
        # Resize img2 to match img1 if needed
        if img1.shape != img2.shape:
            from skimage.transform import resize
            img2 = (resize(img2, img1.shape[:2], preserve_range=True)).astype(np.uint8)
        return img1.astype(np.uint8), img2.astype(np.uint8), 'skimage.astronaut', 'skimage.camera'
    except Exception:
        pass
    
    # Fallback: synthetic images (gradient + checkerboard)
    h, w = 256, 256
    # gradient
    x = np.linspace(0, 1, w)
    grad = np.tile((x*255).astype(np.uint8), (h,1))
    grad_rgb = np.stack([grad, grad, grad], axis=2)
    # checkerboard
    cb = np.indices((h,w)).sum(axis=0) % 2
    cb = (cb * 255).astype(np.uint8)
    cb_rgb = np.stack([cb, cb, cb], axis=2)
    return grad_rgb, cb_rgb, 'synthetic.gradient', 'synthetic.checkerboard'


def to_float_img(img):
    # convert uint8 [0,255] to float64 centered [0,1]
    return img.astype(np.float64) / 255.0

def from_float_img(imgf):
    # clip and convert back to uint8
    imgf = np.clip(imgf, 0.0, 1.0)
    return (imgf * 255.0).astype(np.uint8)

def fft2_image(img):
    # img: HxWxC (C=1 or 3)
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    F = np.zeros((H, W, C), dtype=np.complex128)
    for c in range(C):
        F[..., c] = np.fft.fft2(img[..., c])
    return F

def ifft2_image(F):
    H, W, C = F.shape
    img = np.zeros((H, W, C), dtype=np.float64)
    for c in range(C):
        img[..., c] = np.fft.ifft2(F[..., c]).real
    if img.shape[2] == 1:
        img = img[..., 0]
    return img

def swap_phase(F1, F2):
    # Given complex spectra F1 and F2, swap their phases and return new complex spectra
    mag1 = np.abs(F1)
    mag2 = np.abs(F2)
    ph1 = np.angle(F1)
    ph2 = np.angle(F2)
    new1 = mag1 * np.exp(1j * ph2)
    new2 = mag2 * np.exp(1j * ph1)
    return new1, new2

# --- Main ---
from tkinter import filedialog as fd
img1_np, img2_np, src1, src2 = load_or_fallback(path1 = fd.askopenfilename(), path2= fd.askopenfilename())

# Ensure same size (resize img2 to img1 if needed)
if img1_np.shape != img2_np.shape:
    from PIL import Image as PILImage
    img2_np = np.array(PILImage.fromarray(img2_np).resize((img1_np.shape[1], img1_np.shape[0])))

# Convert to float images in [0,1]
img1 = to_float_img(img1_np)
img2 = to_float_img(img2_np)

# Compute FFT per channel
F1 = fft2_image(img1)
F2 = fft2_image(img2)

# Swap phases
F1s, F2s = swap_phase(F1, F2)

# Reconstruct images (inverse FFT)
rec1 = ifft2_image(F1s)
rec2 = ifft2_image(F2s)

# Convert back to uint8
rec1_u8 = from_float_img(rec1)
rec2_u8 = from_float_img(rec2)
orig1_u8 = (img1_np.astype(np.uint8))
orig2_u8 = (img2_np.astype(np.uint8))

# Display each image in its own matplotlib figure
plt.figure(figsize=(6,6))
if orig1_u8.ndim == 2:
    plt.imshow(orig1_u8, cmap='gray')
else:
    plt.imshow(orig1_u8)
plt.title(f'Original 1 ({src1})')
plt.axis('off')

plt.figure(figsize=(6,6))
if orig2_u8.ndim == 2:
    plt.imshow(orig2_u8, cmap='gray')
else:
    plt.imshow(orig2_u8)
plt.title(f'Original 2 ({src2})')
plt.axis('off')

plt.figure(figsize=(6,6))
if rec1_u8.ndim == 2:
    plt.imshow(rec1_u8, cmap='gray')
else:
    plt.imshow(rec1_u8)
plt.title('Reconstruction: magnitude(image1) + phase(image2)')
plt.axis('off')

plt.figure(figsize=(6,6))
if rec2_u8.ndim == 2:
    plt.imshow(rec2_u8, cmap='gray')
else:
    plt.imshow(rec2_u8)
plt.title('Reconstruction: magnitude(image2) + phase(image1)')
plt.axis('off')

plt.show()