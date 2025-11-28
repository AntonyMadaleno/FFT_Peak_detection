import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from FFT_utils import compute_fft_shift_mag
from Radial_utils import radial_distance_profile

def filter_fft_by_radial_profile(image_path, percentile_top=90, k_sigma=1.0):
    """
    Filtre une image en FFT basé sur le profil radial.
    
    Paramètres:
    -----------
    image_path : str
        Chemin vers l'image à traiter
    percentile_top : float
        Percentile pour définir le seuil de préservation (90 = top 10%)
    k_sigma : float
        Coefficient pour le seuil (médiane + k_sigma * sigma)
    
    Retour:
    -------
    dict contenant toutes les données intermédiaires et résultats
    """
    
    # 1. Charger l'image
    img = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    img_array = np.array(img, dtype=np.float64)
    
    print(f"Image chargée: {img_array.shape}")
    
    # 2. Calculer FFT et FFT shift
    Fsh, mag, mag_log = compute_fft_shift_mag(np, img_array)
    
    print(f"FFT calculée, magnitude shape: {mag.shape}")
    
    # 3. Calculer le profil radial
    radii, mean_profile, counts = radial_distance_profile(mag)
    
    print(f"Profil radial calculé: {len(radii)} rayons")
    
    # 4. Calculer le seuil (médiane + k*sigma du profil)
    valid_profile = mean_profile[mean_profile > 0]
    median_profile = np.median(valid_profile)
    sigma_profile = np.std(valid_profile)
    threshold_profile = median_profile + k_sigma * sigma_profile
    
    print(f"Médiane profil: {median_profile:.2f}")
    print(f"Sigma profil: {sigma_profile:.2f}")
    print(f"Seuil profil: {threshold_profile:.2f}")
    
    # 5. Calculer le seuil pour le top percentile des magnitudes
    flat_mag = mag.ravel()
    threshold_top = np.percentile(flat_mag, percentile_top)
    
    print(f"Seuil top {percentile_top}%: {threshold_top:.2f}")
    
    # 6. Créer le masque de filtrage
    ny, nx = mag.shape
    cy, cx = ny // 2, nx // 2
    
    # Calculer la distance de chaque pixel au centre
    y_coords, x_coords = np.ogrid[:ny, :nx]
    distances = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2).astype(int)

    max_radius = len(mean_profile) - 1
    distances = np.clip(distances, 0, max_radius)
    
    # Créer le masque: garder si profil radial >= seuil OU magnitude dans top percentile
    mask = np.ones((ny, nx), dtype=bool)
    
    for r in range(len(mean_profile)):
        if mean_profile[r] < threshold_profile:
            # Ce rayon devrait être filtré
            mask[distances == r] = False
    
    # Exception: garder les magnitudes dans le top percentile
    mask_top = mag >= threshold_top
    mask = mask | mask_top
    
    pixels_filtered = np.sum(~mask)
    pixels_preserved_by_exception = np.sum((~(mean_profile[distances] >= threshold_profile)) & mask_top)
    
    print(f"Pixels filtrés: {pixels_filtered} ({100*pixels_filtered/(ny*nx):.2f}%)")
    print(f"Pixels préservés par exception (top {percentile_top}%): {pixels_preserved_by_exception}")
    
    # 7. Appliquer le masque à la FFT shiftée
    Fsh_filtered = Fsh.copy()
    Fsh_filtered[~mask] = 0
    
    mag_filtered = np.abs(Fsh_filtered)
    mag_log_filtered = np.log1p(mag_filtered)
    
    # 8. Reconstruction (inverse FFT shift puis inverse FFT)
    F_filtered = np.fft.ifftshift(Fsh_filtered)
    img_reconstructed = np.fft.ifft2(F_filtered)
    img_reconstructed = np.real(img_reconstructed)
    
    # Normaliser pour affichage
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    
    print("Reconstruction terminée")
    
    # Retourner toutes les données
    return {
        'original': img_array,
        'fft_magnitude': mag,
        'fft_magnitude_log': mag_log,
        'fft_filtered_magnitude': mag_filtered,
        'fft_filtered_magnitude_log': mag_log_filtered,
        'reconstructed': img_reconstructed,
        'mask': mask,
        'radial_profile': mean_profile,
        'radii': radii,
        'threshold_profile': threshold_profile,
        'threshold_top': threshold_top,
        'median_profile': median_profile,
        'sigma_profile': sigma_profile,
        'stats': {
            'pixels_filtered': pixels_filtered,
            'pixels_preserved': pixels_preserved_by_exception,
            'filter_ratio': pixels_filtered / (ny * nx)
        }
    }


def visualize_results(results):
    """
    Visualise tous les résultats du filtrage.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Image originale
    ax1 = plt.subplot(3, 3, 1)
    plt.imshow(results['original'], cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    # FFT magnitude (log)
    ax2 = plt.subplot(3, 3, 2)
    plt.imshow(results['fft_magnitude_log'], cmap='viridis')
    plt.title('FFT Magnitude (log)')
    plt.axis('off')
    
    # FFT magnitude filtrée (log)
    ax3 = plt.subplot(3, 3, 3)
    plt.imshow(results['fft_filtered_magnitude_log'], cmap='viridis')
    plt.title('FFT Filtrée (log)')
    plt.axis('off')
    
    # Image reconstruite
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(results['reconstructed'], cmap='gray')
    plt.title('Image Reconstruite')
    plt.axis('off')
    
    # Masque de filtrage
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(results['mask'], cmap='RdYlGn')
    plt.title('Masque (Vert=Conservé, Rouge=Filtré)')
    plt.axis('off')
    
    # Différence
    ax6 = plt.subplot(3, 3, 6)
    diff = np.abs(results['original'] - results['reconstructed'])
    plt.imshow(diff, cmap='hot')
    plt.title('Différence (|Original - Reconstruit|)')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # Profil radial
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(results['radii'], results['radial_profile'], 'b-', linewidth=2, label='Profil radial')
    plt.axhline(y=results['threshold_profile'], color='r', linestyle='--', 
                linewidth=2, label=f"Seuil (μ+σ) = {results['threshold_profile']:.2f}")
    plt.axhline(y=results['median_profile'], color='g', linestyle=':', 
                linewidth=2, label=f"Médiane = {results['median_profile']:.2f}")
    plt.xlabel('Rayon (pixels)')
    plt.ylabel('Somme des magnitudes')
    plt.title('Profil Radial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistiques
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    stats_text = f"""
    STATISTIQUES
    
    Médiane profil: {results['median_profile']:.2f}
    Sigma profil: {results['sigma_profile']:.2f}
    Seuil profil: {results['threshold_profile']:.2f}
    
    Seuil top magnitudes: {results['threshold_top']:.2f}
    
    Pixels filtrés: {results['stats']['pixels_filtered']}
    Ratio filtré: {results['stats']['filter_ratio']*100:.2f}%
    
    Pixels préservés (exception): 
    {results['stats']['pixels_preserved']}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # Histogramme des magnitudes
    ax9 = plt.subplot(3, 3, 9)
    mag_flat = results['fft_magnitude'].ravel()
    plt.hist(mag_flat, bins=100, color='blue', alpha=0.7, label='Original')
    plt.axvline(x=results['threshold_top'], color='r', linestyle='--', 
                linewidth=2, label=f'Top 90% = {results["threshold_top"]:.2f}')
    plt.xlabel('Magnitude')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Magnitudes FFT')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Exemple d'utilisation
if __name__ == "__main__":
    from tkinter import filedialog as fd
    # Remplacer par le chemin de votre image
    image_path = "votre_image.jpg"
    
    # Traiter l'image
    results = filter_fft_by_radial_profile(
        fd.askopenfilename(), 
        percentile_top=95,  # Top 10% préservé
        k_sigma=2.33         # Seuil = médiane + 1*sigma
    )
    
    # Visualiser les résultats
    visualize_results(results)
    
    # Sauvegarder l'image reconstruite
    img_result = Image.fromarray(results['reconstructed'].astype(np.uint8))
    img_result.save("image_filtered.png")
    print("Image filtrée sauvegardée: image_filtered.png")