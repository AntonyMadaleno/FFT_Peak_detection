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
    mean_profile[valid] = sums[valid] # / counts[valid]
    radii = np.arange(len(mean_profile))

    """# Création d'une figure et d'un axe
    fig, ax = plt.subplots(figsize=(6, 3))

    # Tracer sur l'axe
    ax.plot(radii, mean_profile / (ny * nx), lw=1.0)

    # Nom des axes
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized magnitude")

    # Affichage de la figure
    plt.show()

    # Si besoin, fermer explicitement la figure
    plt.close(fig)"""

    return radii, mean_profile, counts

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

def _fit_gmm_with_weights(x, weights, n_components, covariance_type='full', random_state=0, rng_seed=None):
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    try:
        gm.fit(x, sample_weight=weights)
        return gm
    except TypeError:
        w = np.asarray(weights, dtype=float)
        total = w.sum()
        if total <= 0 or np.all(w == 0):
            gm.fit(x)
            return gm
        p = w / total
        N = x.shape[0]
        M = int(min(20000, max(1000, 10 * N)))
        rng = np.random.RandomState(rng_seed)
        idxs = rng.choice(N, size=M, replace=True, p=p)
        sample_x = x[idxs]
        gm.fit(sample_x)
        return gm
    except Exception:
        raise

def _choose_k_by_elbow_on_f(f_vals):
    f = np.asarray(f_vals, dtype=float)
    if not np.all(np.isfinite(f)):
        finite = f[np.isfinite(f)]
        penalty = 1e12 if finite.size == 0 else max(1e6, 10.0 * finite.max())
        f = np.where(np.isfinite(f), f, penalty)
    K = f.size
    if K == 1:
        return 1
    xs = np.arange(1, K+1)
    x1, y1 = xs[0], f[0]
    x2, y2 = xs[-1], f[-1]
    num = np.abs((y2 - y1) * xs - (x2 - x1) * f + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    den = max(den, 1e-12)
    d = num / den
    return int(xs[np.argmax(d)] - 1)

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

def select_peaks_with_gmm_and_components(profile, max_peaks=4, min_distance=1, max_k=None,
                                         covariance_type='full', random_state=0, rng_seed=None,
                                         freq_priority=0.0, freq_cutoff=0.1, freq_exponent=2.0):
    """
    Détecte pics et renvoie :
      sel         : liste de (r_index, profile[r_index]) (au plus max_peaks) - sélection finale par min_distance
      components  : liste de dicts pour chaque composante détectée contenant :
                    { 'mean': float, 'std': float, 'weight': float, 'center_idx': int, 'pdf': np.array,
                      'dirac_added': bool (optionnel), 'dirac_amplitude': float (optionnel), 'is_dirac': bool (optionnel) }
      modeled     : np.array (mixture reconstruite, mise à l'échelle pour conserver l'aire du profile),
                    auquel on ajoute des pics de Dirac détectés directement sur `profile`.
      k_best      : nombre de composantes choisi
      f_vals, Ks  : (optionnel) valeurs f(k) testées et Ks correspondants (utile pour debug)

    Règles de détection des Dirac :
      1) critère principal : profile[idx] > mean + 4.0 * std
      2) si aucun indice retenu par (1), fallback sur discontinuités de la dérivée :
         - d = np.gradient(profile) ; prendre indices où |d| > mean(|d|) + 4.0 * std(|d|)
         - ces indices sont considérés comme positions de discontinuité (candidates)
      - On n'ajoute un Dirac que si aucune composante GMM n'a son centre arrondi à cet indice.
      - Le nombre de Dirac ajoutés est limité à int(max_k/2).
    Priorisation des basses fréquences:
      - freq_priority (float, default 0.0) : force de priorisation (0 = désactivé).
      - freq_cutoff (float in (0,0.5], default 0.1) : fréquence normalisée de coupure (fraction de Nyquist).
      - freq_exponent (float > 0, default 2.0) : contrôle la pente de la transition.
    """
    import numpy as np

    profile = np.asarray(profile, dtype=float)
    N = profile.size
    if N == 0:
        return [], [], np.zeros(0), 0, [], []

    # --- sécurité paramètres fréquence ---
    try:
        freq_cutoff = float(freq_cutoff)
    except Exception:
        freq_cutoff = 0.1
    if freq_cutoff <= 0.0 or freq_cutoff > 0.5:
        freq_cutoff = 0.1
    try:
        freq_priority = float(freq_priority)
    except Exception:
        freq_priority = 0.0
    try:
        freq_exponent = float(freq_exponent)
    except Exception:
        freq_exponent = 2.0
    if freq_exponent <= 0.0:
        freq_exponent = 2.0

    # poids non négatifs pour le fit
    prof_min = profile.min()
    if prof_min < 0:
        weights = profile - prof_min
    else:
        weights = profile.copy()
    if np.all(weights <= 0):
        return [], [], np.zeros(N), 0, [], []

    if max_k is None:
        max_k = min(max_peaks, max(1, N // 2, 6))
    max_k = max(1, int(max_k))

    x = np.arange(N).reshape(-1, 1)

    # Préparer grille fréquentielle (pour rfft)
    freqs = np.fft.rfftfreq(N, d=1.0)  # shape M = N//2 + 1

    # construire poids fréquentiels (prioriser basses fréquences) :
    if freq_priority == 0.0:
        weight_freq = np.ones_like(freqs)
    else:
        denom = 1.0 + (freqs / freq_cutoff) ** freq_exponent
        weight_freq = 1.0 + float(freq_priority) * (1.0 / denom)
    sum_weight_freq = float(np.sum(weight_freq))
    if sum_weight_freq <= 0:
        sum_weight_freq = 1.0

    # calculer f(k) pour k=1..max_k
    f_vals = []
    Ks = list(range(1, max_k + 1))
    for k in Ks:
        try:
            gm = _fit_gmm_with_weights(x, weights, n_components=k,
                                       covariance_type=covariance_type,
                                       random_state=random_state, rng_seed=(rng_seed if rng_seed is not None else random_state))
            log_pdf = gm.score_samples(x)
            pdf = np.exp(log_pdf)
            sum_pdf = pdf.sum()
            sum_profile = profile.sum()
            if sum_pdf <= 0:
                modeled_k = np.zeros_like(profile)
            else:
                modeled_k = (sum_profile / sum_pdf) * pdf

            # si on ne priorise pas les basses fréquences, on conserve dist_Jeffreys d'origine
            if freq_priority == 0.0:
                f_k = dist_Jeffreys(profile, modeled_k)
                if not np.isfinite(f_k):
                    f_k = 1e12
            else:
                # distance spectrale pondérée entre profile et modeled_k
                Pf = np.fft.rfft(profile)
                Mf = np.fft.rfft(modeled_k)
                magP = np.abs(Pf)
                magM = np.abs(Mf)
                diff2 = (magP - magM) ** 2
                weighted_diff = weight_freq * diff2
                spectral_dist = float(np.sum(weighted_diff) / sum_weight_freq)
                f_k = spectral_dist
                if not np.isfinite(f_k):
                    f_k = 1e12

        except Exception:
            f_k = 1e12
        f_vals.append(f_k)

    # choisir k_best selon elbow sur f_vals puis limiter par max_peaks
    k_best = _choose_k_by_elbow_on_f(np.array(f_vals))
    k_best = min(k_best, max_peaks)

    # fit final
    try:
        gm = _fit_gmm_with_weights(x, weights, n_components=k_best,
                                   covariance_type=covariance_type,
                                   random_state=random_state, rng_seed=(rng_seed if rng_seed is not None else random_state))
    except Exception:
        # échec du fit final : retomber proprement
        return [], [], np.zeros(N), 0, f_vals, Ks

    means = gm.means_.reshape(-1)
    variances = _component_variances_from_gm(gm, covariance_type)
    stds = np.sqrt(variances)
    mix_weights = gm.weights_.reshape(-1)

    # construire pdf par composante (non-scaled) : weight_i * N(x|mean,std)
    x1d = np.arange(N)
    comps = []
    eps_var = 1e-12
    safe_variances = np.maximum(variances, eps_var)
    norm_consts = 1.0 / np.sqrt(2.0 * np.pi * safe_variances)
    for i, (m, s, w, normc, var) in enumerate(zip(means, stds, mix_weights, norm_consts, safe_variances)):
        exponent = -0.5 * ((x1d - m) / (s if s > 0 else np.sqrt(var))) ** 2
        comp_pdf = w * normc * np.exp(exponent)   # shape (N,)
        comps.append({'mean': float(m),
                      'std': float(np.sqrt(var)),
                      'weight': float(w),
                      'center_idx': int(np.clip(int(round(m)), 0, N-1)),
                      'pdf_raw': comp_pdf})  # avant scaling

    # mixture raw
    mixture_raw = np.sum([c['pdf_raw'] for c in comps], axis=0) if comps else np.zeros(N)
    sum_profile = profile.sum()
    sum_mixture_raw = mixture_raw.sum()
    if sum_mixture_raw <= 0:
        scale = 0.0
    else:
        scale = sum_profile / sum_mixture_raw
    for c in comps:
        c['pdf'] = scale * c['pdf_raw']
        c['dirac_added'] = False
        c['dirac_amplitude'] = 0.0
        c['is_dirac'] = False

    modeled = scale * mixture_raw

    # tri des composantes par poids de mélange décroissant (pour sélection greedy)
    comps_sorted = sorted(comps, key=lambda cc: cc['weight'], reverse=True)

    # sélection greedy selon min_distance (en indices) en gardant les centres arrondis
    sel = []
    for c in comps_sorted:
        r = c['center_idx']
        val = float(c['pdf'][r])
        too_close = False
        for (sr, sval) in sel:
            if abs(sr - r) <= max(1, min_distance):
                too_close = True
                break
        if not too_close:
            sel.append((r, val))
        if len(sel) >= max_peaks:
            break

    # --- Détection des Dirac sur le profil (critère statistique + fallback dérivée) ---
    profile_mean = float(np.mean(profile)) if profile.size > 0 else 0.0
    profile_std = float(np.std(profile)) if profile.size > 0 else 0.0
    # seuil principal : mean + 4.0 * sigma
    magnitude_threshold = profile_mean + 4.0 * profile_std

    dirac_limit = max(0, int(max_k // 2))
    dirac_added_count = 0

    # indices déjà occupés par une composante GMM (centres arrondis)
    gmm_centers = {c['center_idx'] for c in comps}

    def respects_min_distance(idx, selection, mind):
        for (sr, sval) in selection:
            if abs(sr - idx) <= max(1, mind):
                return False
        return True

    # --- première passe : indices où profile[idx] > magnitude_threshold ---
    candidate_indices = [i for i, val in enumerate(profile) if val > magnitude_threshold]
    candidate_indices = sorted(candidate_indices, key=lambda ii: profile[ii], reverse=True)

    # --- si aucun candidat trouvé, fallback sur discontinuités de la dérivée ---
    if len(candidate_indices) == 0:
        # dérivée discrète (gradient, même longueur que profile)
        d = np.gradient(profile)
        absd = np.abs(d)
        mean_absd = float(np.mean(absd))
        std_absd = float(np.std(absd))
        deriv_threshold = mean_absd + 4.0 * std_absd
        # indices où la dérivée a une discontinuité (|d| > seuil)
        candidate_indices = [i for i, val in enumerate(absd) if val > deriv_threshold]
        # trier par amplitude de la discontinuité (décroissant)
        candidate_indices = sorted(candidate_indices, key=lambda ii: absd[ii], reverse=True)

    # parcourir les candidats et ajouter au besoin (en vérifiant centre GMM et min_distance)
    for r in candidate_indices:
        if dirac_added_count >= dirac_limit:
            break
        if r in gmm_centers:
            # on ne veut pas ajouter si un composant GMM est déjà centré ici
            continue

        dirac_amp = float(profile[r])

        # si amplitude trop faible (?) — on peut garder car threshold l'a filtrée
        # vérifier min_distance vs sélection actuelle
        if not respects_min_distance(r, sel, min_distance):
            # si sel plein et le dirac est plus grand que le plus petit élément, essayer de remplacer
            if len(sel) >= max_peaks:
                min_idx = None
                min_val = float('inf')
                for i, (_sr, _sval) in enumerate(sel):
                    if _sval < min_val:
                        min_val = _sval
                        min_idx = i
                if min_idx is not None and dirac_amp > min_val:
                    temp_sel = sel[:min_idx] + sel[min_idx+1:]
                    if respects_min_distance(r, temp_sel, min_distance):
                        sel[min_idx] = (r, float(dirac_amp))
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            if len(sel) < max_peaks:
                sel.append((r, float(dirac_amp)))
            else:
                min_idx = None
                min_val = float('inf')
                for i, (_sr, _sval) in enumerate(sel):
                    if _sval < min_val:
                        min_val = _sval
                        min_idx = i
                if min_idx is not None and dirac_amp > min_val:
                    temp_sel = sel[:min_idx] + sel[min_idx+1:]
                    if respects_min_distance(r, temp_sel, min_distance):
                        sel[min_idx] = (r, float(dirac_amp))
                    else:
                        continue

        # ajout au modeled
        modeled[r] += dirac_amp

        # créer pdf du Dirac (zéros sauf à r)
        dirac_pdf = np.zeros(N, dtype=float)
        dirac_pdf[r] = dirac_amp

        # ajouter une entrée distincte dans comps représentant le Dirac (std = 0.0)
        dirac_comp = {
            'mean': float(r),
            'std': 0.0,
            'weight': 0.0,
            'center_idx': int(r),
            'pdf': dirac_pdf,
            'dirac_added': True,
            'dirac_amplitude': float(dirac_amp),
            'is_dirac': True
        }
        comps.append(dirac_comp)
        gmm_centers.add(r)
        dirac_added_count += 1

    # trier sel par magnitude décroissante et garder au plus max_peaks
    sel = sorted(sel, key=lambda t: t[1], reverse=True)[:max_peaks]

    return sel, comps, modeled, k_best, f_vals, Ks



