import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

def choose_two_images():
    filenames = filedialog.askopenfilenames(
        title="Choisir deux images",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
    )
    
    if len(filenames) != 2:
        messagebox.showerror("Erreur", "Vous devez choisir exactement 2 images.")
        return
    
    img1_path, img2_path = filenames
    combine_images(img1_path, img2_path)

def combine_images(img1_path, img2_path):
    # Chargement des images
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # Redimensionnement si tailles différentes
    if img1.size != img2.size:
        messagebox.showinfo("Info", "Les images n'ont pas la même taille. "
                                    "La deuxième image sera redimensionnée.")
        img2 = img2.resize(img1.size)

    # Conversion en tableau numpy
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Calcul de (img1 + img2) / 2
    combined_arr = ((arr1 + 0.5 * arr2) / 1.5).astype(np.uint8)

    result_img = Image.fromarray(combined_arr)

    # Choisir l'emplacement de sauvegarde
    save_path = filedialog.asksaveasfilename(
        title="Sauvegarder l'image",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
    )

    if save_path:
        result_img.save(save_path)
        messagebox.showinfo("Succès", f"Image sauvegardée :\n{save_path}")

# Interface Tkinter minimale
root = tk.Tk()
root.title("Combinaison de deux images")
root.geometry("300x150")

btn = tk.Button(root, text="Choisir 2 images", command=choose_two_images)
btn.pack(expand=True)

root.mainloop()
