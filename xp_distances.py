try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import numpy as np

# Configuration
epsilon = 10**(-9)

def get_array_module(arr):
    """Return the appropriate array module (cupy or numpy) based on input type."""
    if GPU_AVAILABLE and hasattr(arr, '__cuda_array_interface__'):
        return cp
    return np

def to_gpu(arr):
    """Convert array to GPU if available, otherwise return as-is."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)

def to_cpu(arr):
    """Convertit récursivement en NumPy (supporte CuPy arrays, tuples, listes, dicts)."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr.get()

    if isinstance(arr, (tuple, list)):
        return type(arr)(to_cpu(x) for x in arr)

    if isinstance(arr, dict):
        return {k: to_cpu(v) for k, v in arr.items()}

    return np.asarray(arr)



def prepare_arrays(P, Q):
    """
    Prepare arrays for computation, handling both single spectra and lists of spectra.
    Returns: (P_arr, Q_arr, is_batch, xp)
    """
    P_arr = to_gpu(P)
    Q_arr = to_gpu(Q)
    xp = get_array_module(P_arr)
    
    # Convert to arrays if not already
    P_arr = xp.asarray(P_arr)
    Q_arr = xp.asarray(Q_arr)
    
    # Handle different input cases
    if P_arr.ndim == 1 and Q_arr.ndim == 1:
        # Single spectrum vs single spectrum
        is_batch = False
        P_arr = P_arr[None, :]  # Add batch dimension
        Q_arr = Q_arr[None, :]
    elif P_arr.ndim == 2 and Q_arr.ndim == 1:
        # Multiple spectra vs single spectrum
        is_batch = True
        Q_arr = xp.broadcast_to(Q_arr[None, :], P_arr.shape)
    elif P_arr.ndim == 1 and Q_arr.ndim == 2:
        # Single spectrum vs multiple spectra
        is_batch = True
        P_arr = xp.broadcast_to(P_arr[None, :], Q_arr.shape)
    elif P_arr.ndim == 2 and Q_arr.ndim == 2:
        # Multiple spectra vs multiple spectra
        is_batch = True
        if P_arr.shape[0] != Q_arr.shape[0]:
            raise ValueError("Number of spectra must match")
    else:
        raise ValueError("Invalid input dimensions")
    
    return P_arr, Q_arr, is_batch, xp

def compute_distance(func, P, Q, *args, **kwargs):
    """Generic wrapper for distance computation with batch support."""
    P_arr, Q_arr, is_batch, xp = prepare_arrays(P, Q)
    
    result = func(P_arr, Q_arr, xp, *args, **kwargs)
    
    if not is_batch:
        result = result[0]  # Return scalar for single spectrum case
    
    return to_cpu(result)

# Minkowski Based Measurements ##################################################################
def _dist_Minkowski(P, Q, xp, k=2):
    return (xp.sum(xp.abs(P - Q)**k, axis=1))**(1/k)

def dist_Minkowski(P, Q, k=2):
    return compute_distance(_dist_Minkowski, P, Q, k)

def _dist_Manhattan(P, Q, xp):
    return xp.sum(xp.abs(P - Q), axis=1)

def dist_Manhattan(P, Q):
    return compute_distance(_dist_Manhattan, P, Q)

def _dist_Euclidienne(P, Q, xp):
    return xp.sqrt(xp.sum((P - Q)**2, axis=1))

def dist_Euclidienne(P, Q):
    return compute_distance(_dist_Euclidienne, P, Q)

def _dist_Chebyshev(P, Q, xp):
    return xp.max(xp.abs(P - Q), axis=1)

def dist_Chebyshev(P, Q):
    return compute_distance(_dist_Chebyshev, P, Q)

def _dist_Sorensen(P, Q, xp):
    return xp.sum(xp.abs(P - Q), axis=1) / xp.sum(P + Q, axis=1)

def dist_Sorensen(P, Q):
    return compute_distance(_dist_Sorensen, P, Q)

def _dist_Soergel(P, Q, xp):
    return xp.sum(xp.abs(P - Q), axis=1) / xp.sum(xp.maximum(P, Q), axis=1)

def dist_Soergel(P, Q):
    return compute_distance(_dist_Soergel, P, Q)

def _dist_Kulczynski_d(P, Q, xp):
    return xp.sum(xp.abs(P - Q), axis=1) / xp.sum(xp.minimum(P, Q), axis=1)

def dist_Kulczynski_d(P, Q):
    return compute_distance(_dist_Kulczynski_d, P, Q)

def _dist_Canberra(P, Q, xp):
    return xp.sum(xp.abs(P - Q) / (P + Q), axis=1)

def dist_Canberra(P, Q):
    return compute_distance(_dist_Canberra, P, Q)

def _dist_Lorentzian(P, Q, xp):
    return xp.sum(xp.log(1 + xp.abs(P - Q)), axis=1)

def dist_Lorentzian(P, Q):
    return compute_distance(_dist_Lorentzian, P, Q)

# Intersection ####################################################################################
def _sim_Intersection(P, Q, xp):
    return xp.sum(xp.minimum(P, Q), axis=1)

def sim_Intersection(P, Q):
    return compute_distance(_sim_Intersection, P, Q)

def _dist_Intersection(P, Q, xp):
    return xp.sum(xp.abs(P - Q), axis=1) / 2

def dist_Intersection(P, Q):
    return compute_distance(_dist_Intersection, P, Q)

# Wave Hedges #####################################################################################
def _Wave_Hedges_1(P, Q, xp):
    min_val = xp.minimum(P, Q)
    max_val = xp.maximum(P, Q)
    return xp.sum(1 - (min_val / max_val), axis=1)

def Wave_Hedges_1(P, Q):
    return compute_distance(_Wave_Hedges_1, P, Q)

def _Wave_Hedges_2(P, Q, xp):
    max_val = xp.maximum(P, Q)
    return xp.sum(xp.abs(P - Q) / max_val, axis=1)

def Wave_Hedges_2(P, Q):
    return compute_distance(_Wave_Hedges_2, P, Q)

# Czekanowski ######################################################################################
def _sim_Czekanowski(P, Q, xp):
    min_val = xp.minimum(P, Q)
    return xp.sum(min_val, axis=1) / xp.sum(P + Q, axis=1)

def sim_Czekanowski(P, Q):
    return compute_distance(_sim_Czekanowski, P, Q)

# Motyka ###########################################################################################
def _sim_Motyka(P, Q, xp):
    min_val = xp.minimum(P, Q)
    return xp.sum(min_val, axis=1) / xp.sum(xp.abs(P + Q), axis=1)

def sim_Motyka(P, Q):
    return compute_distance(_sim_Motyka, P, Q)

def _dist_Motyka(P, Q, xp):
    max_val = xp.maximum(P, Q)
    return xp.sum(max_val, axis=1) / xp.sum(xp.abs(P + Q), axis=1)

def dist_Motyka(P, Q):
    return compute_distance(_dist_Motyka, P, Q)

# Kulczynski s #####################################################################################
def _sim_Kulczynski_s(P, Q, xp):
    min_val = xp.minimum(P, Q)
    return xp.sum(min_val, axis=1) / xp.sum(xp.abs(P - Q), axis=1)

def sim_Kulczynski_s(P, Q):
    return compute_distance(_sim_Kulczynski_s, P, Q)

def _dist_Kulczynski_s(P, Q, xp):
    min_val = xp.minimum(P, Q)
    return xp.sum(xp.abs(P - Q), axis=1) / xp.sum(min_val, axis=1)

def dist_Kulczynski_s(P, Q):
    return compute_distance(_dist_Kulczynski_s, P, Q)

# Ruzicka ##########################################################################################
def _sim_Ruzicka(P, Q, xp):
    min_val = xp.minimum(P, Q)
    max_val = xp.maximum(P, Q)
    return xp.sum(min_val, axis=1) / xp.sum(max_val, axis=1)

def sim_Ruzicka(P, Q):
    return compute_distance(_sim_Ruzicka, P, Q)

# Tani-moto ########################################################################################
def _Tani_moto_1(P, Q, xp):
    min_val = xp.minimum(P, Q)
    sum_P = xp.sum(P, axis=1)
    sum_Q = xp.sum(Q, axis=1)
    sum_min = xp.sum(min_val, axis=1)
    up = sum_P + sum_Q - 2 * sum_min
    bt = sum_P + sum_Q - sum_min
    return up / bt

def Tani_moto_1(P, Q):
    return compute_distance(_Tani_moto_1, P, Q)

def _Tani_moto_2(P, Q, xp):
    min_val = xp.minimum(P, Q)
    max_val = xp.maximum(P, Q)
    return xp.sum(max_val - min_val, axis=1) / xp.sum(max_val, axis=1)

def Tani_moto_2(P, Q):
    return compute_distance(_Tani_moto_2, P, Q)

# Inner Product #####################################################################################
def _sim_Inner_product(P, Q, xp):
    return xp.sum(P * Q, axis=1)

def sim_Inner_product(P, Q):
    return compute_distance(_sim_Inner_product, P, Q)

# Harmonic mean #####################################################################################
def _sim_Harmonic_mean(P, Q, xp):
    return 2 * xp.sum((P * Q) / (P + Q), axis=1)

def sim_Harmonic_mean(P, Q):
    return compute_distance(_sim_Harmonic_mean, P, Q)

# Cosine ############################################################################################
def _sim_Cosine(P, Q, xp):
    dot_product = xp.sum(P * Q, axis=1)
    norm_P = xp.sqrt(xp.sum(P**2, axis=1))
    norm_Q = xp.sqrt(xp.sum(Q**2, axis=1))
    return dot_product / (norm_P * norm_Q)

def sim_Cosine(P, Q):
    return compute_distance(_sim_Cosine, P, Q)

# Jaccard ###########################################################################################
def _sim_Jaccard(P, Q, xp):
    prod = xp.sum(P * Q, axis=1)
    sum_P2 = xp.sum(P**2, axis=1)
    sum_Q2 = xp.sum(Q**2, axis=1)
    return prod / (sum_P2 + sum_Q2 - prod)

def sim_Jaccard(P, Q):
    return compute_distance(_sim_Jaccard, P, Q)

def _dist_Jaccard(P, Q, xp):
    prod = xp.sum(P * Q, axis=1)
    sum_P2 = xp.sum(P**2, axis=1)
    sum_Q2 = xp.sum(Q**2, axis=1)
    return xp.sum((P - Q)**2, axis=1) / (sum_P2 + sum_Q2 - prod)

def dist_Jaccard(P, Q):
    return compute_distance(_dist_Jaccard, P, Q)

# Dice ###############################################################################################
def _sim_Dice(P, Q, xp):
    prod = xp.sum(P * Q, axis=1)
    sum_P2 = xp.sum(P**2, axis=1)
    sum_Q2 = xp.sum(Q**2, axis=1)
    return (2 * prod) / (sum_P2 + sum_Q2)

def sim_Dice(P, Q):
    return compute_distance(_sim_Dice, P, Q)

def _dist_Dice(P, Q, xp):
    sum_P2 = xp.sum(P**2, axis=1)
    sum_Q2 = xp.sum(Q**2, axis=1)
    return xp.sum((P - Q)**2, axis=1) / (sum_P2 + sum_Q2)

def dist_Dice(P, Q):
    return compute_distance(_dist_Dice, P, Q)

# Bhattacharyya ######################################################################################
def _dist_Bhattacharyya(P, Q, xp):
    return -xp.log(xp.sum(xp.sqrt(P * Q), axis=1))

def dist_Bhattacharyya(P, Q):
    return compute_distance(_dist_Bhattacharyya, P, Q)

# Hellinger ##########################################################################################
def _dist_Hellinger_1(P, Q, xp):
    return xp.sqrt(2 * xp.sum((xp.sqrt(P) - xp.sqrt(Q))**2, axis=1))

def dist_Hellinger_1(P, Q):
    return compute_distance(_dist_Hellinger_1, P, Q)

def _dist_Hellinger_2(P, Q, xp):
    return 2 * xp.sqrt(1 - xp.sum(xp.sqrt(P * Q), axis=1))

def dist_Hellinger_2(P, Q):
    return compute_distance(_dist_Hellinger_2, P, Q)

# Matusita ###########################################################################################
def _dist_Matusita_1(P, Q, xp):
    return xp.sqrt(xp.sum((xp.sqrt(P) - xp.sqrt(Q))**2, axis=1))

def dist_Matusita_1(P, Q):
    return compute_distance(_dist_Matusita_1, P, Q)

def _dist_Matusita_2(P, Q, xp):
    return xp.sqrt(2 - 2 * xp.sum(xp.sqrt(P * Q), axis=1))

def dist_Matusita_2(P, Q):
    return compute_distance(_dist_Matusita_2, P, Q)

# Squared-chord ######################################################################################
def _dist_Squared_chord(P, Q, xp):
    return xp.sum((xp.sqrt(P) - xp.sqrt(Q))**2, axis=1)

def dist_Squared_chord(P, Q):
    return compute_distance(_dist_Squared_chord, P, Q)

def _sim_Squared_chord(P, Q, xp):
    return 2 * xp.sum(xp.sqrt(P * Q), axis=1) - 1

def sim_Squared_chord(P, Q):
    return compute_distance(_sim_Squared_chord, P, Q)

# Squared Euclidean ##################################################################################
def _dist_Squared_Euclidean(P, Q, xp):
    return xp.sum((P - Q)**2, axis=1)

def dist_Squared_Euclidean(P, Q):
    return compute_distance(_dist_Squared_Euclidean, P, Q)

# Pearson chi square #################################################################################
def _dist_Pearson_chi(P, Q, xp):
    return xp.sum((P - Q)**2 / Q, axis=1)

def dist_Pearson_chi(P, Q):
    return compute_distance(_dist_Pearson_chi, P, Q)

# Neyman chi #########################################################################################
def _dist_Neyman_chi(P, Q, xp):
    return xp.sum((P - Q)**2 / P, axis=1)

def dist_Neyman_chi(P, Q):
    return compute_distance(_dist_Neyman_chi, P, Q)

# chi Square #########################################################################################
def _dist_Squared_chi(P, Q, xp):
    return xp.sum((P - Q)**2 / (P + Q), axis=1)

def dist_Squared_chi(P, Q):
    return compute_distance(_dist_Squared_chi, P, Q)

# Divergence #########################################################################################
def _Divergence(P, Q, xp):
    return 2 * xp.sum((P - Q)**2 / (P + Q)**2, axis=1)

def Divergence(P, Q):
    return compute_distance(_Divergence, P, Q)

# Clark ##############################################################################################
def _dist_Clark(P, Q, xp):
    return xp.sqrt(xp.sum((xp.abs(P - Q) / (P + Q))**2, axis=1))

def dist_Clark(P, Q):
    return compute_distance(_dist_Clark, P, Q)

# Additive Symmetric chi square ######################################################################
def _Additive_symmetric_chi(P, Q, xp):
    return xp.sum(((P - Q)**2 * (P + Q)) / (P * Q), axis=1)

def Additive_symmetric_chi(P, Q):
    return compute_distance(_Additive_symmetric_chi, P, Q)

# Kullback-Leibler ###################################################################################
def _dist_KL(P, Q, xp):
    Qe = xp.where(Q == 0, epsilon, Q)
    Pe = xp.where(P == 0, epsilon, P)
    Qe = Qe / xp.sum(Qe)
    Pe = Pe / xp.sum(Pe)
    return xp.sum(P * xp.log(Pe / Qe), axis=1)

def dist_KL(P, Q):
    return compute_distance(_dist_KL, P, Q)

# Kullback-Leibler Pseudo Divergence ##################################################################

def _dist_KLPD(S1, S2, xp):
    ep = 1e-6
    min_val = 1e-12

    # Replace zeros to avoid division by zero
    S1 = xp.where(S1 == 0, ep, S1)
    S2 = xp.where(S2 == 0, ep, S2)

    # Compute integrals
    k1 = xp.trapz(S1, axis=-1)
    k2 = xp.trapz(S2, axis=-1)

    # Normalize spectra
    if S1.ndim > 1:
        k1_exp = k1[..., xp.newaxis]
        k2_exp = k2[..., xp.newaxis]
    else:
        k1_exp = k1
        k2_exp = k2

    N1 = S1 / k1_exp + ep
    N2 = S2 / k2_exp + ep

    # Ratio with clipping
    ratio_N = xp.clip(N1 / N2, min_val, None)
    ratio_N_inv = xp.clip(N2 / N1, min_val, None)
    ratio_k = xp.clip(k1 / k2, min_val, None)

    p1 = N1 * xp.log(ratio_N)
    p2 = N2 * xp.log(ratio_N_inv)

    G = k1 * xp.trapz(p1, axis=-1) + k2 * xp.trapz(p2, axis=-1)
    W = (k1 - k2) * xp.log(ratio_k)

    return G, W

def dist_KLPD(P, Q):
    return compute_distance(_dist_KLPD, P, Q)

def dist_KLPD_Sum(P, Q):
    G, W = compute_distance(_dist_KLPD, P, Q)
    return G + W

def dist_KLPD_Energy(P, Q):
    G, W = compute_distance(_dist_KLPD, P, Q)
    return W

def dist_KLPD_Shape(P, Q):
    G, W = compute_distance(_dist_KLPD, P, Q)
    return G

# Jeffreys ###########################################################################################
def _dist_Jeffreys(P, Q, xp):
    return (_dist_KL(P, Q, xp) + _dist_KL(Q, P, xp)) / 2.0

def dist_Jeffreys(P, Q):
    return compute_distance(_dist_Jeffreys, P, Q)

# K divergence #######################################################################################
def _K_divergence(P, Q, xp):
    Qe = xp.where(Q == 0, epsilon, Q)
    return xp.sum(P * xp.log((2 * P) / (P + Qe)), axis=1)

def K_divergence(P, Q):
    return compute_distance(_K_divergence, P, Q)

# Topsoe #############################################################################################
def _dist_Topsoe(P, Q, xp):
    Qe = xp.where(Q == 0, epsilon, Q)
    return xp.sum(P * xp.log((2 * P) / (Qe + P)) + Q * xp.log((2 * Q) / (Qe + P)), axis=1)

def dist_Topsoe(P, Q):
    return compute_distance(_dist_Topsoe, P, Q)

# Jensen-Shannon #####################################################################################
def _dist_Jensen_Shannon(P, Q, xp):
    return (_K_divergence(P, Q, xp) + _K_divergence(Q, P, xp)) / 2

def dist_Jensen_Shannon(P, Q):
    return compute_distance(_dist_Jensen_Shannon, P, Q)

# Jensen difference #####################################################################################
def _delta_Jensen(P, Q, xp):
    return xp.sum((P + xp.log(P) + Q + xp.log(Q)) / 2 - 
                  ((P + Q) / 2) * xp.log((P + Q) / 2), axis=1)

def delta_Jensen(P, Q):
    return compute_distance(_delta_Jensen, P, Q)

# Utility function for checking GPU availability
def is_gpu_available():
    """Check if GPU computation is available."""
    return GPU_AVAILABLE

def get_device_info():
    """Get information about the current computational device."""
    if GPU_AVAILABLE:
        return f"GPU available: {cp.cuda.runtime.getDeviceCount()} device(s)"
    else:
        return "GPU not available, using CPU"