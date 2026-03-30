"""
models/fem_bridge.py — FEM Bridge Module
Pipeline : séries temporelles pont → fréquences propres (FFT)
→ matrice de rigidité K → diagnostic NOVA-Ω style
Dataset : schéma ponts norvégiens open-access (Zenodo 10507957)
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import warnings; warnings.filterwarnings("ignore")
np.random.seed(42)

BRIDGE_PARAMS = {
    "name":           "Norwegian Bridge Reference (Bergsøysund schema)",
    "length_m":       931,
    "n_sensors":      12,
    "fs_hz":          10.0,
    "n_modes":        6,
    "freq_nominal":   [0.18, 0.31, 0.47, 0.68, 0.89, 1.12],
    "damping_pct":    [1.2,  1.5,  1.8,  2.1,  2.3,  2.5],
    "mass_modal_kg":  5e5,
}
SENSOR_LABELS = [
    "ACC_MID_V","ACC_MID_L","ACC_MID_T",
    "ACC_Q1_V","ACC_Q1_L","ACC_Q3_V","ACC_Q3_L",
    "STRAIN_1","STRAIN_2","STRAIN_3",
    "TEMP_DECK","WIND_SPEED",
]

def generate_bridge_data(T_seconds=3600, damage_level=0.0, seed=0):
    np.random.seed(seed)
    fs   = BRIDGE_PARAMS["fs_hz"]
    N    = int(T_seconds * fs)
    t    = np.arange(N) / fs
    freqs_nom = np.array(BRIDGE_PARAMS["freq_nominal"])
    damp      = np.array(BRIDGE_PARAMS["damping_pct"]) / 100
    freqs     = freqs_nom * (1.0 - 0.5 * damage_level)
    data      = np.zeros((N, 12))

    for i in range(6):
        amp   = 1.0 / (i+1) * (1 - 0.2*damage_level)
        phi   = 2*np.pi*freqs[i]*t + np.random.uniform(0, np.pi)
        modal = amp * np.exp(-damp[i]*2*np.pi*freqs[i]*t*0.001) * np.sin(phi)
        shapes = [np.sin((i+1)*np.pi*s) for s in [0.5,0.25,0.75,0.25,0.5,0.75,0.5]]
        for j in range(7):
            data[:,j] += shapes[j] * modal
    for j in range(7):
        data[:,j] += 0.015 * np.random.randn(N)
    for j in range(3):
        data[:,7+j] = 0.4*(data[:,j]**2)*(1+2*damage_level) + 0.008*np.random.randn(N)
    data[:,10] = 15 + 5*np.sin(2*np.pi*t/86400) + 0.5*np.random.randn(N)
    data[:,11] = 8  + 3*np.sin(2*np.pi*t/3600)  + 1.0*np.random.randn(N)
    return data, t

def extract_modal_params(data, fs=10.0, n_modes=6):
    N,D   = data.shape
    fft_avg = np.zeros(N//2+1)
    for j in range(7):
        fft_avg += np.abs(np.fft.rfft(data[:,j])) / 7.0
    freq_axis = np.fft.rfftfreq(N, d=1/fs)
    smooth    = savgol_filter(fft_avg[:len(freq_axis)], 15, 3)
    df        = freq_axis[1] - freq_axis[0]
    min_dist  = max(1, int(0.05/df))
    peaks, _  = find_peaks(smooth,
                            height=np.percentile(smooth,75),
                            distance=min_dist)
    tops      = sorted(peaks, key=lambda p: smooth[p], reverse=True)[:n_modes]
    freqs_found = sorted([float(freq_axis[p]) for p in tops])
    while len(freqs_found) < n_modes:
        freqs_found.append(freqs_found[-1]*1.3 if freqs_found else 0.5)
    return {
        "frequencies_hz": freqs_found[:n_modes],
        "freq_axis":      freq_axis.tolist(),
        "fft_spectrum":   smooth.tolist(),
    }

def build_stiffness_matrix(modal_params, n_dof=6):
    freqs  = np.array(modal_params["frequencies_hz"][:n_dof])
    mass   = BRIDGE_PARAMS["mass_modal_kg"]
    omega2 = (2*np.pi*freqs)**2
    k_diag = mass * omega2          # N/m — unités physiques réelles

    K = np.diag(k_diag.astype(float))
    for i in range(n_dof-1):
        k_off = -0.30 * np.sqrt(k_diag[i]*k_diag[i+1])
        K[i, i+1] = k_off
        K[i+1, i] = k_off
    K += 1e-3 * np.eye(n_dof)
    return K

def diagnose_matrix(K, freq_nominal=None, freq_measured=None):
    n       = K.shape[0]
    eigvals = np.linalg.eigvalsh(K)
    pos_eig = eigvals[eigvals > 0]
    kappa   = pos_eig[-1]/pos_eig[0] if len(pos_eig)>=2 else np.inf
    sym_err = np.max(np.abs(K-K.T)) / (np.max(np.abs(K))+1e-10)
    n_neg   = int(np.sum(eigvals < 0))
    diag_dom= all(abs(K[i,i])>=sum(abs(K[i,j]) for j in range(n) if j!=i)
                  for i in range(n))

    # Chute de fréquence par rapport au nominal
    freq_drop_pct = 0.0
    if freq_nominal is not None and freq_measured is not None:
        nom = np.array(freq_nominal[:len(freq_measured)])
        mea = np.array(freq_measured[:len(nom)])
        freq_drop_pct = float(np.mean((nom-mea)/nom*100))

    diagnostics = []
    severity    = "NOMINAL"

    if n_neg > 0:
        diagnostics.append({
            "pattern":"Negative Eigenvalues","severity":"CRITICAL",
            "description":f"{n_neg} negative eigenvalue(s) — not positive definite",
            "cause":"Structural instability or severe local damage",
            "action":"Immediate inspection — do not operate",
        }); severity="CRITICAL"

    if freq_drop_pct > 12:
        diagnostics.append({
            "pattern":"Significant Frequency Drop","severity":"HIGH",
            "description":f"Modal frequencies dropped {freq_drop_pct:.1f}% vs nominal",
            "cause":"Substantial stiffness loss — cracking or connection failure",
            "action":"Structural inspection within 48 hours",
        }); severity = "HIGH" if severity=="NOMINAL" else severity
    elif freq_drop_pct > 5:
        diagnostics.append({
            "pattern":"Moderate Frequency Shift","severity":"MEDIUM",
            "description":f"Modal frequencies dropped {freq_drop_pct:.1f}% vs nominal",
            "cause":"Early-stage stiffness degradation — possible fatigue or settlement",
            "action":"Schedule inspection within 30 days",
        });
        if severity=="NOMINAL": severity="MEDIUM"

    if kappa > 1e6:
        diagnostics.append({
            "pattern":"Near-Singular Stiffness","severity":"HIGH",
            "description":f"Condition number κ = {kappa:.2e}",
            "cause":"Local stiffness loss — solver convergence at risk",
            "action":"Review structural connections and boundary conditions",
        });
        if severity=="NOMINAL": severity="HIGH"

    if sym_err > 0.05:
        diagnostics.append({
            "pattern":"Asymmetric Stiffness","severity":"MEDIUM",
            "description":f"Symmetry error = {sym_err:.4f}",
            "cause":"Asymmetric damage or sensor inconsistency",
            "action":"Verify sensor calibration",
        });
        if severity=="NOMINAL": severity="MEDIUM"

    if not diag_dom:
        diagnostics.append({
            "pattern":"Poor DOF Coupling","severity":"LOW",
            "description":"Matrix not diagonally dominant",
            "cause":"Strong inter-DOF coupling",
            "action":"Review boundary conditions",
        });
        if severity=="NOMINAL": severity="LOW"

    if not diagnostics:
        diagnostics.append({
            "pattern":"Nominal","severity":"NOMINAL",
            "description":"Well-conditioned, positive definite — structure healthy",
            "cause":"Normal operation","action":"Continue regular monitoring",
        })

    return {
        "severity":severity,"kappa":float(kappa),"n_dof":n,
        "n_neg_eigvals":n_neg,"sym_error":float(sym_err),
        "diag_dominant":diag_dom,"freq_drop_pct":freq_drop_pct,
        "min_eigval":float(eigvals[0]),"max_eigval":float(eigvals[-1]),
        "diagnostics":diagnostics,"K":K,"eigvals":eigvals.tolist(),
    }

def extract_modal_params_guided(data, fs=10.0, n_modes=6, freq_range=(0.05, 3.0)):
    """
    Extraction guidée : cherche uniquement dans freq_range.
    Plus robuste que la version aveugle pour les signaux courts.
    """
    N, D = data.shape
    fft_avg = np.zeros(N//2+1)
    for j in range(min(7, D)):
        fft_avg += np.abs(np.fft.rfft(data[:, j])) / min(7, D)
    freq_axis = np.fft.rfftfreq(N, d=1/fs)
    smooth    = savgol_filter(fft_avg[:len(freq_axis)], 11, 3)

    # Masquer hors de la plage physique
    mask = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
    smooth_masked = smooth.copy()
    smooth_masked[~mask] = 0

    df       = freq_axis[1] - freq_axis[0]
    min_dist = max(1, int(0.08/df))
    peaks, _ = find_peaks(smooth_masked,
                           height=np.percentile(smooth_masked[mask], 60),
                           distance=min_dist)
    tops     = sorted(peaks, key=lambda p: smooth_masked[p], reverse=True)[:n_modes]
    freqs_found = sorted([float(freq_axis[p]) for p in tops])
    while len(freqs_found) < n_modes:
        freqs_found.append(freqs_found[-1]*1.3 if freqs_found else 0.5)
    return {
        "frequencies_hz": freqs_found[:n_modes],
        "freq_axis":      freq_axis.tolist(),
        "fft_spectrum":   smooth.tolist(),
    }
