"""
Synthetic Grayscale Image Generator (Tkinter GUI)

Features:
- Patterns: sinusoid, checker, lines, circles, hexagons
- Noise: gaussian, poisson
- Resolution: width x height
- Orientation parameter for patterns (degrees)
- GPU acceleration with CuPy when available; falls back to NumPy. If the requested image has < 1,000,000 pixels, CPU (NumPy) will be used even if GPU is toggled ON.
- Display embedded via matplotlib in Tkinter
- Save generated image (uses OpenCV if available, otherwise Pillow)

Run: python synthetic_image_generator.py
Dependencies: numpy, matplotlib, pillow, opencv-python (optional), cupy (optional)
"""

import sys
import math
import traceback

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    print("Tkinter is required to run this application.")
    raise

# GPU fallback: try CuPy; otherwise use NumPy
import numpy as _np
try:
    import cupy as _cp
    GPU_AVAILABLE = True
    xp_default = _cp
except Exception:
    _cp = None
    GPU_AVAILABLE = False
    xp_default = _np

# Matplotlib for display (embedded in Tk)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# OpenCV for saving (optional); fallback to Pillow
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False
    from PIL import Image


# ----------------------------- Utilities -----------------------------

def xp_asnumpy(arr, xp):
    """Return numpy array for display / saving. Handles cupy arrays."""
    if xp is _cp and _cp is not None:
        return _cp.asnumpy(arr)
    return arr


def ensure_uint8(arr):
    """Convert a float image in [0,1] or arbitrary range to uint8 0-255."""
    arr = _np.asarray(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        return (_np.clip(arr, 0, 1) * 255).astype(_np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (_np.clip(norm, 0, 1) * 255).astype(_np.uint8)


# ----------------------------- Pattern Generators -----------------------------

def make_normalized_grid(width, height, xp):
    # Coordinates in pixel units [0..width-1], [0..height-1]
    x = xp.linspace(0, width - 1, width)
    y = xp.linspace(0, height - 1, height)
    xv, yv = xp.meshgrid(x, y)
    return xv, yv


def rotate_coords(xv, yv, cx, cy, angle_deg, xp):
    # rotate around center (cx,cy) by angle_deg
    theta = angle_deg * math.pi / 180.0
    cosa = xp.cos(theta)
    sina = xp.sin(theta)
    xr = (xv - cx) * cosa - (yv - cy) * sina
    yr = (xv - cx) * sina + (yv - cy) * cosa
    return xr + cx, yr + cy


def pattern_sinusoid(width, height, size, orientation_deg, xp):
    """Sinusoid applied along x-axis then rotated by orientation.

    Important semantics changed per user request:
    - `size` is now the **wavelength** in pixels (longueur d'onde).
      e.g. size=20 => the sinusoid repeats every 20 pixels.
    - Orientation rotates the pattern after creating it along the x-axis.
    """
    xv, yv = make_normalized_grid(width, height, xp)
    # center coords for rotation
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    # rotate coordinates by -orientation so that original pattern along x becomes rotated in output
    xr, yr = rotate_coords(xv, yv, cx, cy, -orientation_deg, xp)
    # wavelength in pixels
    wavelength = max(1.0, float(size))
    img = 0.5 * (1.0 + xp.sin(2.0 * xp.pi * (xr / wavelength)))
    return img


def pattern_checker(width, height, size, xp):
    # size here is diameter/spacing in pixels (for checker-like patterns we use spacing=size)
    xv, yv = make_normalized_grid(width, height, xp)
    p = max(1.0, float(size))
    cx = _np.floor(xv / p) % 2
    cy = _np.floor(yv / p) % 2
    img = xp.where((cx + cy) % 2 == 0, 1.0, 0.0)
    return img


def pattern_lines(width, height, size, orientation_deg, xp):
    # size: spacing in pixels between lines. orientation rotates the vertical lines.
    xv, yv = make_normalized_grid(width, height, xp)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    xr, yr = rotate_coords(xv, yv, cx, cy, -orientation_deg, xp)
    # lines along x (i.e., vertical stripes before rotation): use modulus with spacing
    spacing = max(1.0, float(size))
    val = 0.5 * (1.0 + xp.sin(2.0 * xp.pi * (xr / spacing)))
    # smooth and normalize
    img = (val - val.min()) / (val.max() - val.min() + 1e-12)
    return img


def pattern_circles(width, height, size, xp):
    # size: diameter in pixels; centers spaced by size (tile like checker)
    xv, yv = make_normalized_grid(width, height, xp)
    d = max(1.0, float(size))
    # compute distance to center of each tile cell
    rx = xp.abs((xv % d) - d/2.0)
    ry = xp.abs((yv % d) - d/2.0)
    dist = xp.sqrt(rx**2 + ry**2)
    img = xp.where(dist <= (d / 2.0), 1.0, 0.0)
    return img


def pattern_hexagons(width, height, size, xp):
    # Hexagon centers spaced by size, hexagons diameter = size
    xv, yv = make_normalized_grid(width, height, xp)
    s = max(1.0, float(size))
    # vertical spacing between hexagon rows (approximate for tight packing)
    v_spacing = s * (3.0 / 4.0)
    # compute grid indices for candidate centers
    col = xp.floor(xv / s)
    row = xp.floor(yv / v_spacing)
    # center positions (shift every other row)
    cx = (col + 0.5 * (row % 2)) * s + s / 2.0
    cy = row * v_spacing + s / 2.0
    dx = xv - cx
    dy = yv - cy
    # conservative hexagon test using an anisotropic box metric approximating hex shape
    a = xp.abs(dx)
    b = xp.abs(dy) * (2.0 / math.sqrt(3.0))
    metric = xp.maximum(a, b)
    img = xp.where(metric <= (s / 2.0), 1.0, 0.0)
    return img


def generate_pattern(kind: str, width: int, height: int, size: float, orientation_deg: float, xp=None):
    """Dispatch to the pattern generator.
    semantics:
      - if kind == 'sinusoid' : size is wavelength in pixels (longueur d'onde).
      - for checker, lines, circles, hexagons: size is interpreted in pixels (diameter / spacing)
    """
    if xp is None:
        xp = xp_default
    kind = kind.lower()
    if kind == "sinusoid":
        return pattern_sinusoid(width, height, size, orientation_deg, xp)
    if kind == "checker":
        return pattern_checker(width, height, size, xp)
    if kind == "lines":
        return pattern_lines(width, height, size, orientation_deg, xp)
    if kind == "circles":
        return pattern_circles(width, height, size, xp)
    if kind == "hexagons":
        return pattern_hexagons(width, height, size, xp)
    raise ValueError(f"Unknown pattern: {kind}")


# ----------------------------- Noise -----------------------------

def add_gaussian_noise(img, intensity, xp):
    # intensity interpreted as sigma in fraction of dynamic range (0..1)
    sigma = float(intensity)
    if xp is _cp and _cp is not None:
        noise = xp.random.standard_normal(img.shape) * sigma
    else:
        noise = _np.random.standard_normal(img.shape) * sigma
    noisy = img + noise
    return noisy


def add_poisson_noise(img, intensity, xp):
    # intensity acts as "peak counts" multiplier: higher -> stronger Poisson
    peak = max(1.0, float(intensity) * 100.0)
    arr = img
    if xp is _cp and _cp is not None:
        counts = xp.clip(arr, 0, 1) * peak
        noisy_counts = xp.random.poisson(counts)
        noisy = noisy_counts / peak
    else:
        counts = _np.clip(arr, 0, 1) * peak
        noisy_counts = _np.random.poisson(counts)
        noisy = noisy_counts / peak
    return noisy


def apply_noise(img, noise_type: str, intensity: float, xp):
    noise_type = noise_type.lower()
    if noise_type == "none":
        return img
    if noise_type == "gaussian":
        return add_gaussian_noise(img, intensity, xp)
    if noise_type == "poisson":
        return add_poisson_noise(img, intensity, xp)
    raise ValueError(f"Unknown noise type: {noise_type}")


# ----------------------------- GUI Application -----------------------------

class SyntheticImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthetic Grayscale Image Generator")
        self.geometry("1100x720")

        self.xp = xp_default
        self.gpu_on = GPU_AVAILABLE

        # Defaults
        self.pattern_var = tk.StringVar(value="sinusoid")
        self.size_var = tk.DoubleVar(value=8.0)
        self.orientation_var = tk.DoubleVar(value=0.0)  # degrees
        self.noise_var = tk.StringVar(value="none")
        self.noise_intensity = tk.DoubleVar(value=0.05)
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)

        self._build_ui()

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.current_image = None  # numpy array uint8

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.display_frame = ttk.Frame(self)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Pattern selection
        ttk.Label(ctrl, text="Pattern:").pack(anchor=tk.W)
        pattern_menu = ttk.Combobox(ctrl, textvariable=self.pattern_var, state='readonly')
        pattern_menu['values'] = ("sinusoid", "checker", "lines", "circles", "hexagons")
        pattern_menu.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="Size / Wavelength:").pack(anchor=tk.W)
        # user requested an entry rather than a slider
        size_entry = ttk.Entry(ctrl, textvariable=self.size_var)
        size_entry.pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, text="(Sinusoid: wavelength in pixels, e.g. size=20 -> period 20px. Others: diameter/spacing in pixels)").pack()

        ttk.Label(ctrl, text="Orientation (deg):").pack(anchor=tk.W, pady=(8,0))
        orient_entry = ttk.Entry(ctrl, textvariable=self.orientation_var)
        orient_entry.pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)

        # Noise
        ttk.Label(ctrl, text="Noise type:").pack(anchor=tk.W)
        noise_menu = ttk.Combobox(ctrl, textvariable=self.noise_var, state='readonly')
        noise_menu['values'] = ("none", "gaussian", "poisson")
        noise_menu.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="Noise intensity:").pack(anchor=tk.W)
        noise_slider = ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.noise_intensity, orient=tk.HORIZONTAL)
        noise_slider.pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, textvariable=self.noise_intensity).pack()

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)

        # Resolution
        ttk.Label(ctrl, text="Width:").pack(anchor=tk.W)
        w_entry = ttk.Entry(ctrl, textvariable=self.width_var)
        w_entry.pack(fill=tk.X, pady=2)
        ttk.Label(ctrl, text="Height:").pack(anchor=tk.W)
        h_entry = ttk.Entry(ctrl, textvariable=self.height_var)
        h_entry.pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)

        # GPU toggle / info
        gpu_frame = ttk.Frame(ctrl)
        gpu_frame.pack(fill=tk.X, pady=4)
        self.gpu_label = ttk.Label(gpu_frame, text=("GPU ON" if self.gpu_on else "GPU not available"))
        self.gpu_label.pack(side=tk.LEFT)
        toggle_gpu = ttk.Button(gpu_frame, text="Toggle GPU", command=self.toggle_gpu)
        toggle_gpu.pack(side=tk.RIGHT)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=6)

        # Buttons
        gen_btn = ttk.Button(ctrl, text="Generate", command=self.generate_and_display)
        gen_btn.pack(fill=tk.X, pady=4)

        save_btn = ttk.Button(ctrl, text="Save Image...", command=self.save_image)
        save_btn.pack(fill=tk.X, pady=4)

        help_btn = ttk.Button(ctrl, text="About / Dependencies", command=self.show_about)
        help_btn.pack(fill=tk.X, pady=4)

    def toggle_gpu(self):
        if not GPU_AVAILABLE:
            messagebox.showinfo("GPU", "CuPy not found. GPU acceleration not available.")
            return
        # Toggle between xp=_cp and xp=_np
        if self.xp is _cp:
            self.xp = _np
            self.gpu_on = False
            self.gpu_label.config(text="GPU OFF")
        else:
            self.xp = _cp
            self.gpu_on = True
            self.gpu_label.config(text="GPU ON")

    def generate_and_display(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers")
            kind = self.pattern_var.get()
            size = float(self.size_var.get())
            orientation = float(self.orientation_var.get())
            noise = self.noise_var.get()
            intensity = float(self.noise_intensity.get())

            # If image is small (< 1,000,000 pixels) force CPU
            if width * height < 1_000_000:
                xp = _np
            else:
                xp = self.xp

            # Generate
            img = generate_pattern(kind, width, height, size, orientation, xp=xp)
            img = apply_noise(img, noise, intensity, xp)

            # Ensure numpy array for display
            arr = xp_asnumpy(img, xp)
            uint8 = ensure_uint8(arr)
            self.current_image = uint8

            # Display
            self.ax.clear()
            self.ax.axis('off')
            self.ax.imshow(uint8, cmap='gray', vmin=0, vmax=255)
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def save_image(self):
        if self.current_image is None:
            messagebox.showinfo("Save", "No image generated yet. Please generate first.")
            return
        filetypes = [("PNG image", "*.png"), ("TIFF image", "*.tif *.tiff"), ("All files", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=filetypes)
        if not path:
            return
        try:
            if OPENCV_AVAILABLE:
                cv2.imwrite(path, self.current_image)
            else:
                im = Image.fromarray(self.current_image)
                im.save(path)
            messagebox.showinfo("Saved", f"Image saved to {path}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Save error", str(e))

    def show_about(self):
        deps = []
        deps.append(f"NumPy: {_np.__version__}")
        deps.append(f"CuPy: {getattr(_cp, '__version__', 'not installed')}")
        deps.append(f"OpenCV: {'installed' if OPENCV_AVAILABLE else 'not installed'}")
        messagebox.showinfo("About / Dependencies", deps)


if __name__ == '__main__':
    app = SyntheticImageApp()
    app.mainloop()
