"""
fft_magnitude_save.py

Ouvre une image via un filedialog tkinter, calcule la FFT 2D (shifted),
applique log(1+|F|) et sauvegarde l'image du spectre avec cmap='magma'.
Dépendances : numpy, scipy, pillow, matplotlib
"""

import sys
import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

def load_image_as_gray(path):
    """Charge l'image et renvoie un array float (grayscale, type float32)."""
    img = Image.open(path).convert("RGB")  # garde RGB, on convertit ensuite en gris
    arr = np.asarray(img, dtype=np.float32)
    # conversion pondérée standard (luminosity) -> gris
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = arr.astype(np.float32)
    return gray

def compute_log_magnitude_spectrum(image_array):
    """Calcule fft2, shift, magnitude et applique log1p, puis normalise 0..1."""
    F = fft2(image_array)
    Fshift = fftshift(F)
    magnitude = np.abs(Fshift)
    log_mag = np.log1p(magnitude)  # log(1 + |F|)
    # normalisation min=0, max=1
    minv = log_mag.min()
    maxv = log_mag.max()
    if maxv - minv > 0:
        norm = (log_mag - minv) / (maxv - minv)
    else:
        norm = np.zeros_like(log_mag)
    return norm**(2.2)

def main():
    root = tk.Tk()
    root.withdraw()  # cacher la fenêtre principale

    # Ouvrir fichier image
    filetypes = [("Images", ("*.ppm","*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")), ("All files", "*.*")]
    img_path = filedialog.askopenfilename(title="Choisir une image", filetypes=filetypes)
    if not img_path:
        messagebox.showinfo("Annulé", "Aucun fichier sélectionné. Fin.")
        return

    try:
        img = load_image_as_gray(img_path)
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{e}")
        return

    # Calcul FFT et spectre normalisé
    spectrum = compute_log_magnitude_spectrum(img)

    # Choisir où sauvegarder
    save_path = filedialog.asksaveasfilename(
        title="Sauvegarder le spectre (FFT magnitude log)",
        defaultextension=".png",
        filetypes=[("PNG image","*.png"), ("JPEG image","*.jpg;*.jpeg"), ("TIFF image","*.tif;*.tiff")]
    )
    if not save_path:
        messagebox.showinfo("Annulé", "Sauvegarde annulée.")
        return

    try:
        # plt.imsave permet d'appliquer cmap directement
        plt.imsave(save_path, spectrum, cmap='magma', format=None)  # format déduit de l'extension
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de sauvegarder le fichier:\n{e}")
        return

    messagebox.showinfo("Terminé", f"Sauvegarde réussie :\n{save_path}")

if __name__ == "__main__":
    main()