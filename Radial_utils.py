import math
import numpy as np
from sklearn.mixture import GaussianMixture
from xp_distances import dist_Jeffreys
import matplotlib.pyplot as plt

# ---------------- radial profile utilities ----------------

def radial_distance_profile(mag, center=None, max_radius=None):
    ny, nx = mag.shape
    if center is None:
        cy, cx = ny//2, nx//2
    else:
        cy, cx = int(center[0]), int(center[1])
    if max_radius is None:
        max_radius = min(cy, cx, ny-cy-1, nx-cx-1)
    max_radius = int(max_radius)
    sums = np.zeros(max_radius+1, dtype=np.float64)
    counts = np.zeros(max_radius+1, dtype=np.float64)
    for r in range(ny):
        dy = r - cy
        for c in range(nx):
            dx = c - cx
            rad_float = math.hypot(dy, dx)
            rad_ceil = int(np.ceil(rad_float))
            rad_floor = int(np.floor(rad_float))

            if rad_floor <= max_radius:
                sums[rad_floor] += mag[r, c] * ( 1 - np.abs(rad_float - float(rad_floor)) )
                counts[rad_floor] += ( 1 - np.abs(rad_float - float(rad_ceil)) )

            if rad_ceil <= max_radius:
                sums[rad_ceil] += mag[r, c] * ( 1 - np.abs(rad_float - float(rad_ceil)) )
                counts[rad_ceil] += ( 1 - np.abs(rad_float - float(rad_ceil)) )
                
    mean_profile = np.zeros_like(sums)
    valid = counts > 0
    mean_profile[valid] = (sums[valid] / counts[valid]) / (nx * ny)
    P = np.mean(mean_profile)
    Db_profile = 10 * np.log10( (mean_profile / P) + 1)
    radii = np.arange(len(mean_profile))

    sym_radii = np.concatenate([-radii[::-1], radii[1:]])  # [-N+1, ..., -1, 0, 1, ..., N-1]

    # Symétrisation du Db_profile
    sym_Db_profile = np.concatenate([Db_profile[::-1], Db_profile[1:]])

    return radii, Db_profile, counts, sym_radii, sym_Db_profile

def angle_profile(mag, angle, center=None, radial_resolution=1.0, angular_tolerance=1.0):
    """
    Profil des magnitudes en fonction de la distance pour une direction/angle fixé.

    Paramètres
    ----------
    mag : 2D array-like
        Image de magnitudes (ny, nx).
    angle : float
        Angle cible en degrés (0..360). 0 = direction +x (à droite), angle croissant anti-horaire.
    center : tuple (y, x) ou None
        Centre de coordonnées (par défaut centre de l'image).
    radial_resolution : float
        Largeur d'un bin radial en pixels (>= 0.1 recommandé).
    angular_tolerance : float
        Demi-largeur en degrés de la fenêtre angulaire autour de `angle` :
        seul les pixels avec |Δangle| <= angular_tolerance contribuent. Pondération linéaire.

    Retour
    -------
    distances : 1D numpy array (float)
        Centres des bins radiaux (en pixels) : (i + 0.5) * radial_resolution
    magnitudes : 1D numpy array (float)
        Profil des magnitudes (moyenne pondérée par bin). Zéro si aucun pixel.
    """
    if radial_resolution <= 0:
        raise ValueError("radial_resolution must be > 0")
    if angular_tolerance <= 0:
        raise ValueError("angular_tolerance must be > 0")
    mag = np.asarray(mag)
    if mag.ndim != 2:
        raise ValueError("mag must be a 2D array")

    ny, nx = mag.shape
    if center is None:
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = int(center[0]), int(center[1])

    # rayon maximal possible (entier de bins)
    max_radius = min(cy, cx, ny - cy - 1, nx - cx - 1)
    if max_radius <= 0:
        return np.array([]), np.array([])

    # nombre de bins radiaux
    n_bins = int(math.ceil(float(max_radius) / float(radial_resolution)))
    if n_bins < 1:
        n_bins = 1

    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    # normaliser angle cible dans [0,360)
    target_angle = float(angle) % 360.0

    for r in range(ny):
        dy = r - cy
        for c in range(nx):
            dx = c - cx
            rad = math.hypot(dy, dx)
            if rad == 0.0:
                # angle indéfini au centre ; on peut ignorer ou définir arbitrairement (ici on ignore)
                continue
            # angle pixel en degrés [0,360)
            pix_angle = math.degrees(math.atan2(dy, dx)) % 360.0

            # différence angulaire minimale (signe dans [-180,180])
            dtheta = pix_angle - target_angle
            # wrap to [-180,180]
            if dtheta > 180.0:
                dtheta -= 360.0
            elif dtheta < -180.0:
                dtheta += 360.0

            abs_dtheta = abs(dtheta)
            if abs_dtheta > angular_tolerance:
                continue  # pixel hors fenêtre angulaire

            # pondération angulaire linéaire (1 à centre, 0 à la tolérance)
            ang_w = 1.0 - (abs_dtheta / float(angular_tolerance))

            # position radiale dans l'échelle des bins (ex: rad / radial_resolution = 12.34)
            rad_pos = rad / float(radial_resolution)
            floor_idx = int(math.floor(rad_pos))
            frac = rad_pos - math.floor(rad_pos)
            ceil_idx = floor_idx + 1

            val = float(mag[r, c])

            # contribution partagée entre deux bins radiaux (si dans l'intervalle)
            # poids total = ang_w
            w_floor = (1.0 - frac) * ang_w
            w_ceil = frac * ang_w

            if 0 <= floor_idx < n_bins:
                sums[floor_idx] += val * w_floor
                counts[floor_idx] += w_floor
            if 0 <= ceil_idx < n_bins:
                sums[ceil_idx] += val * w_ceil
                counts[ceil_idx] += w_ceil

    magnitudes = np.zeros_like(sums)
    nonzero = counts > 0.0
    magnitudes[nonzero] = sums[nonzero] / counts[nonzero]

    distances = (np.arange(n_bins) + 0.5) * float(radial_resolution)
    return distances, magnitudes

def direction_profile(mag, distance, center=None, resolution=1.0):
    """
    Calcule le profil de magnitude en fonction de la direction pour une distance fixée.

    Arguments
    ---------
    mag : 2D array-like
        Image de magnitudes (shape: ny, nx).
    distance : float
        Distance radiale (en pixels) à analyser.
    center : tuple (y, x) or None
        Centre depuis lequel calculer les angles et distances. Par défaut centre de l'image.
    resolution : float
        Résolution angulaire en degrés (ex: 1.0). Doit être > 0.

    Retour
    ------
    angles : 1D numpy array (dtype=float)
        Centres des bins angulaires en degrés (valeurs dans [0, 360)).
    magnitudes : 1D numpy array (dtype=float)
        Profil des magnitudes (moyenne pondérée dans chaque bin).
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    mag = np.asarray(mag)
    if mag.ndim != 2:
        raise ValueError("mag must be a 2D array")
    ny, nx = mag.shape

    if center is None:
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = int(center[0]), int(center[1])

    # nombre de bins angulaires
    n_bins = max(1, int(math.ceil(360.0 / float(resolution))))

    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    for r in range(ny):
        dy = r - cy
        for c in range(nx):
            dx = c - cx
            rad = math.hypot(dy, dx)
            # pondération en fonction de la proximité à la distance demandée
            radial_weight = 1.0 - abs(rad - float(distance))
            if radial_weight <= 0.0:
                continue  # pixel trop éloigné de la distance d'intérêt

            # angle en degrés dans [0, 360)
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0.0:
                angle += 360.0

            # position dans l'échelle des bins (ex: angle/resolution = 12.34)
            angle_pos = angle / float(resolution)
            floor_idx = int(math.floor(angle_pos)) % n_bins
            frac = angle_pos - math.floor(angle_pos)
            ceil_idx = (floor_idx + 1) % n_bins

            # distribution linéaire entre les deux bins adjacents
            w_floor = (1.0 - frac) * radial_weight
            w_ceil = frac * radial_weight

            val = float(mag[r, c])
            sums[floor_idx] += val * w_floor
            counts[floor_idx] += w_floor
            sums[ceil_idx] += val * w_ceil
            counts[ceil_idx] += w_ceil

    # moyenne pondérée par bin (0 si aucun poids)
    magnitudes = np.zeros_like(sums)
    nonzero = counts > 0.0
    magnitudes[nonzero] = sums[nonzero] / counts[nonzero]

    # centres des bins en degrés (dans [0,360) )
    angles = ((np.arange(n_bins) + 0.5) * resolution) % 360.0

    return angles, magnitudes

def find_peaks_1d(profile, min_distance=3, threshold=None, top_n=8):
    N = len(profile)
    candidates = []
    for i in range(1, N-1):
        if profile[i] <= profile[i-1] or profile[i] <= profile[i+1]:
            continue
        if threshold is not None and profile[i] < threshold:
            continue
        candidates.append((i, profile[i]))
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept = []
    used = set()
    for (i, val) in candidates:
        if any(abs(i - u) <= min_distance for u in used): # Strict filtering
            continue
        kept.append((i, val))
        used.add(i)
        if len(kept) >= top_n:
            break
    return kept

def _choose_k_by_elbow_on_f(f_vals):
    f = np.asarray(f_vals, dtype=float)
    if not np.all(np.isfinite(f)):
        finite = f[np.isfinite(f)]
        penalty = 1e12 if finite.size == 0 else max(1e6, 10.0 * finite.max())
        f = np.where(np.isfinite(f), f, penalty)
    K = f.size
    if K == 1:
        return 1
    
    # Normaliser les données pour une meilleure comparaison
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)
    xs = np.arange(1, K+1)
    
    # Calculer la distance perpendiculaire à la ligne reliant le premier et dernier point
    x1, y1 = xs[0], f_norm[0]
    x2, y2 = xs[-1], f_norm[-1]
    num = np.abs((y2 - y1) * xs - (x2 - x1) * f_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    den = max(den, 1e-12)
    d = num / den
    
    # Calculer aussi la courbure locale (différence seconde)
    if K >= 3:
        # Dérivée seconde normalisée
        second_deriv = np.zeros(K)
        for i in range(1, K-1):
            second_deriv[i] = f_norm[i-1] - 2*f_norm[i] + f_norm[i+1]
        second_deriv = np.abs(second_deriv)
        
        # Score combiné : distance + courbure
        combined_score = d * (1 + second_deriv)
    else:
        combined_score = d
    
    # Trouver le point avec le score maximal, en privilégiant les points plus à droite
    # en cas d'égalité (pour éviter de s'arrêter trop tôt)
    max_score = np.max(combined_score)
    threshold = 0.8 * max_score  # Considérer les points à au moins 80% du max
    candidates = np.where(combined_score >= threshold)[0]
    
    # Parmi les candidats, choisir le plus à droite
    chosen_idx = candidates[-1] if candidates.size > 0 else np.argmax(combined_score)
    
    return int(xs[chosen_idx] + 1)

def _component_variances_from_gm(gm, covariance_type):
    """Retourne un array de variances (1D) par composante pour un GMM 1D."""
    if covariance_type == 'full':
        covs = np.array([gm.covariances_[i][0, 0] for i in range(gm.n_components)])
    elif covariance_type == 'tied':
        covs = np.array([gm.covariances_[0, 0]] * gm.n_components)
    elif covariance_type == 'diag':
        covs = np.array(gm.covariances_).reshape(-1)
    elif covariance_type == 'spherical':
        covs = np.array(gm.covariances_).reshape(-1)
    else:
        covs = np.abs(np.array(gm.covariances_).reshape(-1))
    # sécurité: variances positives
    covs = np.maximum(covs, 1e-12)
    return covs

def select_peaks_with_gmm_and_components(
    x, profile,
    max_peaks=16,
    max_k=None,
    covariance_type='full',
    random_state=0,
    rng_seed=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    profile = np.asarray(profile, dtype=float)
    x = np.asarray(x, dtype=float)
    N = profile.size
    if N == 0:
        return [], [], np.zeros(0), 0, [], []

    # poids non négatifs
    prof_min = profile.min()
    weights = profile - prof_min if prof_min < 0 else profile.copy()
    if np.all(weights <= 0):
        return [], [], np.zeros(N), 0, [], []

    if max_k is None:
        max_k = min(max_peaks, max(1, N // 2, 16))
    max_k = max(1, int(max_k))

    rng = np.random.RandomState(rng_seed)

    f_vals = []
    Ks = list(range(1, max_k + 1))

    p = weights / weights.sum()

    """# ----- boucle sur k -----
    for k in Ks:
        # rééchantillonnage pondéré
        M = min(20000, max(1000, 10 * N))
        idx = rng.choice(N, size=M, replace=True, p=p)

        gm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter= 1000
        )
        gm.fit(x[idx].reshape(-1, 1))

        pdf = np.exp(gm.score_samples(x.reshape(-1, 1)))

        if pdf.mean() <= 0:
            modeled_k = np.zeros_like(profile)
        else:
            modeled_k = pdf * (profile.mean() / pdf.mean())

        f_k = np.trapz(np.abs(weights - modeled_k * np.trapz(weights, x)), x)
        if not np.isfinite(f_k):
            f_k = 1e12
        f_vals.append(f_k)

    # ----- choix de k -----
    print(f_vals)
    k_best = np.argmin(f_vals) + 1
    k_best = min(k_best, max_peaks)

    print(k_best)"""

    # ----- fit final -----
    try:
        M = min(20000, max(1000, 10 * N))
        idx = rng.choice(N, size=M, replace=True, p=p)

        gm = GaussianMixture(
            n_components=max_k,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter= 5000
        )
        gm.fit(x[idx].reshape(-1, 1))

        pdf = np.exp(gm.score_samples(x.reshape(-1, 1)))
        modeled = pdf

        # Créer une nouvelle figure explicite
        fig, ax = plt.subplots(figsize=(6, 3))

        # Tracer sur cette figure spécifique
        ax.plot(x, weights, label='Profile')
        #ax.plot(x, modeled, label='Modeled')

        # Labels des axes
        ax.set_xlabel('Relative Scale')
        ax.set_ylabel('Magnitude (dB)')

        # Définir les valeurs que vous voulez afficher
        desired_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

        # Trouver les indices dans x qui sont les plus proches de ces valeurs
        tick_indices = []
        tick_labels = []

        for desired in desired_ticks:
            # Trouver l'indice le plus proche
            idx = np.argmin(np.abs(x - desired))
            tick_indices.append(x[idx])
            tick_labels.append(f'{ (1 - np.abs(x[idx])):.2f}')

        # Appliquer les ticks
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)

        # Légende et grille
        # ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Afficher UNIQUEMENT cette figure
        plt.show(block=True)

    except Exception as e:
        print(e)
        return [], [], np.zeros(N), 0, f_vals, Ks

    means = gm.means_.reshape(-1)
    variances = _component_variances_from_gm(gm, covariance_type)
    stds = np.sqrt(variances)
    mix_weights = gm.weights_.reshape(-1)

    # ----- composantes -----
    comps = []
    eps_var = 1e-12
    safe_variances = np.maximum(variances, eps_var)
    norm_consts = 1.0 / np.sqrt(2.0 * np.pi * safe_variances)

    for m, s, w, normc, var in zip(means, stds, mix_weights, norm_consts, safe_variances):
        exponent = -0.5 * ((x - m) / (s if s > 0 else np.sqrt(var))) ** 2
        comp_pdf = w * normc * np.exp(exponent)
        center_idx = int(np.argmin(np.abs(x - m)))

        comps.append({
            'mean': float(m),
            'std': float(np.sqrt(var)),
            'weight': float(w),
            'center_idx': center_idx,
            'pdf_raw': comp_pdf
        })

    # ----- mixture -----
    mixture_raw = np.sum([c['pdf_raw'] for c in comps], axis=0) if comps else np.zeros(N)

    if mixture_raw.mean() <= 0:
        scale = 0.0
    else:
        scale = profile.mean() / mixture_raw.mean()

    for c in comps:
        c['pdf'] = scale * c['pdf_raw']
        c['dirac_added'] = False
        c['dirac_amplitude'] = 0.0
        c['is_dirac'] = False

    modeled = scale * mixture_raw

    # ----- sélection des pics -----
    comps_sorted = sorted(comps, key=lambda cc: cc['weight'], reverse=True)

    sel = []
    for c in comps_sorted:
        r = c['center_idx']
        val = float(c['pdf'][r])
        sel.append((r, val))
        if len(sel) >= max_peaks:
            break

    sel = sorted(sel, key=lambda t: t[1], reverse=True)[:max_peaks]

    return sel, comps, modeled, max_k, f_vals, Ks
