"""
Fichier de test pour la détection de peaks 1D par GMM.
Contient :
 - l'implémentation de find_peaks_1d_gmm_elbow (sélection par loi du coude sur l'erreur L1)
 - un __main__ qui génère plusieurs profils synthétiques, exécute la détection
   et trace les résultats pour vérification visuelle.

Usage:
    python test_find_peaks_gmm.py

Dépendances:
    numpy, scikit-learn, matplotlib

Remarques sur robustesse :
 - Certaines versions de scikit-learn n'acceptent pas `sample_weight` dans
   GaussianMixture.fit() ; on essaie d'abord d'utiliser sample_weight, et si
   l'appel lève TypeError on retombe sur un échantillonnage pondéré (bootstrap)
   pour approximer l'effet des poids.
 - Les valeurs infinies/NaN sur la courbe f(k) sont gérées en leur attribuant
   une forte pénalité pour éviter les warnings et permettre la détection du coude.

"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def _choose_k_by_elbow_on_f(f_vals):
    """
    Détection du 'knee' (loi du coude) sur la courbe f(k) donnée pour k=1..K.
    Renvoie l'indice k (1-based) correspondant au point d'inflexion via
    distance perpendiculaire à la droite joignant (1,f1) et (K,fK).

    Cette version est robuste aux valeurs inf/nan : on remplace ces valeurs par
    une pénalité très élevée avant le calcul.
    """
    f = np.asarray(f_vals, dtype=float)
    # remplacer inf/nan par une valeur élevée (pénalité)
    if not np.all(np.isfinite(f)):
        finite_mask = np.isfinite(f)
        if np.any(finite_mask):
            max_finite = np.max(f[finite_mask])
            penalty = max(1e6, 10.0 * max_finite)
        else:
            penalty = 1e12
        f = np.where(np.isfinite(f), f, penalty)

    K = f.size
    if K == 1:
        return 1
    xs = np.arange(1, K+1)
    x1, y1 = xs[0], f[0]
    x2, y2 = xs[-1], f[-1]
    # distance perpendiculaire d_i entre (x_i, f_i) et la ligne (x1,y1)-(x2,y2)
    num = np.abs((y2 - y1) * xs - (x2 - x1) * f + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    den = np.maximum(den, 1e-12)
    d = num / den
    k_choice = int(xs[np.argmax(d)])
    return k_choice


def _fit_gmm_with_weights(x, weights, n_components, covariance_type='full',
                          random_state=0, rng_seed=None):
    """
    Tente de fit un GaussianMixture en utilisant sample_weight si disponible;
    sinon retombe sur un échantillonnage pondéré (bootstrap) pour approximer
    l'effet des poids.

    x: array shape (N,1)
    weights: array shape (N,) >= 0
    Retourne l'objet GaussianMixture entraîné.
    """
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                         random_state=random_state)
    # essayer sample_weight d'abord
    try:
        gm.fit(x, sample_weight=weights)
        return gm
    except TypeError:
        # sample_weight non supporté par cette version de sklearn
        # construire un échantillon pondéré par tirage avec prob ~ weights
        w = np.asarray(weights, dtype=float)
        total = w.sum()
        if total <= 0 or np.all(w == 0):
            # fit sans poids
            gm.fit(x)
            return gm
        p = w / total
        # taille d'échantillon raisonnable pour approx : min(20000, max(1000, 10*N))
        N = x.shape[0]
        M = int(min(20000, max(1000, 10 * N)))
        rng = np.random.RandomState(rng_seed)
        idxs = rng.choice(N, size=M, replace=True, p=p)
        sample_x = x[idxs]
        gm.fit(sample_x)
        return gm
    except Exception:
        # autres erreurs -> ré-élever pour être capturé par l'appelant
        raise


def find_peaks_1d_gmm_elbow(profile, min_distance=3, threshold=None, top_n=8,
                            gmm_count=None, max_k=None, random_state=0,
                            covariance_type='full', return_f_values=False):
    """
    Détecte des 'peaks' sur un profil 1D et modélise par GMM.
    Sélection automatique du nombre de composantes via la loi du coude
    appliquée à f(k) = sum |profile - modeled_profile_k| (on cherche le point
    d'inflexion où l'ajout d'une gaussienne n'améliore plus significativement).

    Retour: liste de tuples (mean, std, nearest_index, weight) triée par weight décroissant.
    Si return_f_values=True, retourne (results, f_values, tested_Ks).
    """
    profile = np.asarray(profile, dtype=float)
    N = profile.size
    if N == 0:
        return [] if not return_f_values else ([], [], [])
    # Préparer poids pour le fit : rendre non-négatif
    prof_min = profile.min()
    if prof_min < 0:
        weights_for_fit = profile - prof_min
    else:
        weights_for_fit = profile.copy()
    # threshold
    if threshold is not None:
        weights_for_fit = np.where(profile >= threshold, weights_for_fit, 0.0)
    if np.all(weights_for_fit <= 0):
        return [] if not return_f_values else ([], [], [])
    x = np.arange(N).reshape(-1, 1)
    if max_k is None:
        max_k = min(top_n, max(1, N // 2))
    max_k = max(1, int(max_k))

    rng_seed = int(random_state) if random_state is not None else None

    # Si gmm_count fixé
    if gmm_count is not None and int(gmm_count) >= 1:
        k_best = int(gmm_count)
        Ks_test = [k_best]
        f_values = []
    else:
        Ks_test = list(range(1, max_k + 1))
        f_values = []
        for k in Ks_test:
            try:
                gm = _fit_gmm_with_weights(x, weights_for_fit, n_components=k,
                                           covariance_type=covariance_type,
                                           random_state=random_state, rng_seed=rng_seed)
                log_pdf = gm.score_samples(x)
                pdf = np.exp(log_pdf)
                sum_pdf = pdf.sum()
                sum_profile = profile.sum()
                if sum_pdf <= 0:
                    modeled = np.zeros_like(profile)
                else:
                    scale = (sum_profile / sum_pdf) if sum_pdf > 0 else 0.0
                    modeled = scale * pdf
                f_k = float(np.sum(np.abs(profile - modeled)))
                # sécurité : si f_k nan/inf -> grande pénalité
                if not np.isfinite(f_k):
                    f_k = 1e12
            except Exception:
                f_k = 1e12
            f_values.append(f_k)
        k_best = _choose_k_by_elbow_on_f(f_values)

    # limiter
    k_best = min(k_best, top_n)

    # Fit final
    gm = _fit_gmm_with_weights(x, weights_for_fit, n_components=k_best,
                               covariance_type=covariance_type,
                               random_state=random_state, rng_seed=rng_seed)
    means = gm.means_.reshape(-1)
    # extraire covariances
    if covariance_type == 'full':
        covs = np.array([gm.covariances_[i][0, 0] for i in range(gm.n_components)])
    elif covariance_type == 'tied':
        covs = np.array([gm.covariances_[0, 0]] * gm.n_components)
    elif covariance_type == 'diag':
        covs = gm.covariances_.reshape(-1)
    elif covariance_type == 'spherical':
        covs = gm.covariances_.reshape(-1)
    else:
        covs = np.abs(gm.covariances_.reshape(-1))
    stds = np.sqrt(np.maximum(covs, 1e-12))
    weights_comp = gm.weights_.reshape(-1)
    results = []
    for m, s, w in zip(means, stds, weights_comp):
        idx = int(np.clip(int(round(m)), 0, N-1))
        results.append((float(m), float(s), idx, float(w)))
    results.sort(key=lambda t: t[3], reverse=True)
    if return_f_values:
        if not f_values:
            log_pdf = gm.score_samples(x)
            pdf = np.exp(log_pdf)
            sum_pdf = pdf.sum()
            sum_profile = profile.sum()
            scale = (sum_profile / sum_pdf) if sum_pdf > 0 else 0.0
            modeled = scale * pdf
            f_k = float(np.sum(np.abs(profile - modeled)))
            f_values = [f_k]
            Ks_test = [k_best]
        return results[:top_n], f_values, Ks_test
    else:
        return results[:top_n]


# ------------------------
# Fonctions utilitaires pour tests
# ------------------------

def synthesize_profile(N=200, peaks=None, noise_std=0.05, baseline=0.0, random_state=None):
    """Génère un profil 1D composé de plusieurs gaussiennes.

    peaks: liste de tuples (center, amplitude, sigma)
    """
    rng = np.random.RandomState(random_state)
    x = np.arange(N)
    prof = np.ones(N) * baseline
    if peaks is None:
        peaks = [(50, 1.0, 3.0), (120, 0.8, 6.0)]
    for c, a, s in peaks:
        prof += a * np.exp(-0.5 * ((x - c) / s)**2)
    prof += rng.normal(scale=noise_std, size=N)
    prof = np.clip(prof, a_min=0.0, a_max=None)
    return prof


def plot_profile_and_model(profile, results, title=None):
    N = len(profile)
    x = np.arange(N)
    # reconstruire modèle GMM à partir du nombre de composantes détectées
    k = len(results)
    if k == 0:
        modeled = np.zeros_like(profile)
    else:
        # tenter d'utiliser notre helper _fit_gmm_with_weights pour garder
        # la compatibilité avec différentes versions de scikit-learn
        try:
            gm = _fit_gmm_with_weights(x.reshape(-1, 1), profile, n_components=k)
        except Exception:
            # fallback: fit sans poids
            gm = GaussianMixture(n_components=k)
            gm.fit(x.reshape(-1, 1))
        log_pdf = gm.score_samples(x.reshape(-1, 1))
        pdf = np.exp(log_pdf)
        sum_pdf = pdf.sum()
        sum_profile = profile.sum()
        scale = (sum_profile / sum_pdf) if sum_pdf > 0 else 0.0
        modeled = scale * pdf
    plt.figure(figsize=(8, 3))
    plt.plot(x, profile, label='profile')
    plt.plot(x, modeled, label='GMM modeled')
    for (m, s, idx, w) in results:
        plt.axvline(m, color='k', linestyle='--', linewidth=1)
        plt.text(m, max(profile)*0.9, f"m={m:.1f} sigma={s:.1f}", ha='center')
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()


if __name__ == '__main__':
    # Cas 1: un seul pic
    prof1 = synthesize_profile(N=200, peaks=[(80, 1.0, 5.0)], noise_std=0.02, random_state=1)
    res1, fvals1, Ks1 = find_peaks_1d_gmm_elbow(prof1, top_n=5, return_f_values=True)
    print('== Profil 1 (un pic) ==')
    print('f(k) tested:', list(zip(Ks1, fvals1)))
    print('Results:', res1)
    plot_profile_and_model(prof1, res1, title='Profil 1: un pic')

    # Cas 2: deux pics
    prof2 = synthesize_profile(N=300, peaks=[(70, 1.2, 4.0), (200, 0.9, 8.0)], noise_std=0.03, random_state=2)
    res2, fvals2, Ks2 = find_peaks_1d_gmm_elbow(prof2, top_n=6, return_f_values=True)
    print('== Profil 2 (deux pics) ==')
    print('f(k) tested:', list(zip(Ks2, fvals2)))
    print('Results:', res2)
    plot_profile_and_model(prof2, res2, title='Profil 2: deux pics')

    # Cas 3: plusieurs pics proches
    prof3 = synthesize_profile(N=400, peaks=[(80, 1.0, 4.0), (95, 0.7, 3.0), (180, 1.1, 6.0), (260, 0.6, 4.5)], noise_std=0.04, random_state=3)
    res3, fvals3, Ks3 = find_peaks_1d_gmm_elbow(prof3, top_n=8, return_f_values=True)
    print('== Profil 3 (plusieurs pics proches) ==')
    print('f(k) tested:', list(zip(Ks3, fvals3)))
    print('Results:', res3)
    plot_profile_and_model(prof3, res3, title='Profil 3: plusieurs pics proches')

    # Cas 4: forcer le nombre de composantes
    prof4 = synthesize_profile(N=200, peaks=[(50, 1.0, 3.0), (120, 0.8, 6.0)], noise_std=0.02, random_state=4)
    res4 = find_peaks_1d_gmm_elbow(prof4, gmm_count=1, top_n=5)
    print('== Profil 4 (gmm_count forcé à 1) ==')
    print('Results:', res4)
    plot_profile_and_model(prof4, res4, title='Profil 4: gmm_count=1 forcé')

    plt.show()
