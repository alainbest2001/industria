"""
utils/data_loader.py — INDUSTRIA
Charge les données NASA SMAP/MSL depuis le cache .npz embarqué dans le repo.
Une seule lecture au démarrage, mis en cache Streamlit — zéro téléchargement externe.
"""
import numpy as np
import ast
import os

# ── Chemin du cache ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH = os.path.join(_BASE, "data", "nasa_cache.npz")

# ── Chargement unique en mémoire ───────────────────────────────────────────────
_cache = None

def _load_cache():
    global _cache
    if _cache is None:
        _cache = np.load(_CACHE_PATH, allow_pickle=True)
    return _cache


def list_channels(dataset: str = "SMAP") -> list:
    """Retourne la liste des canaux disponibles pour SMAP ou MSL."""
    cache = _load_cache()
    result = []
    for key in cache.files:
        if key.endswith("_sc"):
            sc = str(cache[key][0])
            chan = key.replace("_sc", "")
            if sc.upper() == dataset.upper():
                result.append(chan)
    return sorted(result)


def load_channel(chan_id: str) -> dict:
    """
    Charge un canal depuis le cache .npz.
    Retourne dict avec : train, test, labels, anomaly_sequences, source, chan_id
    """
    cache = _load_cache()

    train  = cache[f"{chan_id}_train"].astype(np.float32)
    test   = cache[f"{chan_id}_test"].astype(np.float32)

    # Le test contient parfois une colonne label en plus — la retirer
    if test.shape[1] == train.shape[1] + 1:
        test = test[:, :train.shape[1]]

    labels = cache[f"{chan_id}_labels"].astype(np.int32)
    seqs   = cache[f"{chan_id}_seqs"].tolist()

    return {
        "train":             train,
        "test":              test,
        "labels":            labels,
        "anomaly_sequences": seqs,
        "source":            "nasa_cache",
        "chan_id":           chan_id,
    }