import os
from tkinter import Tk, filedialog
from PIL import Image

def convert_ppm_to_png():
    # Initialize Tkinter (no main window)
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Ask user to select a folder
    folder_path = filedialog.askdirectory(title="Select folder containing PPM files")
    if not folder_path:
        print("No folder selected. Aborting.")
        return

    # Iterate over PPM files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".ppm"):
            ppm_path = os.path.join(folder_path, filename)
            png_path = os.path.join(
                folder_path,
                os.path.splitext(filename)[0] + ".png"
            )

            try:
                with Image.open(ppm_path) as img:
                    img.save(png_path, format="PNG")
                print(f"Converted: {filename} -> {os.path.basename(png_path)}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

    print("Conversion completed.")

if __name__ == "__main__":
    convert_ppm_to_png()
