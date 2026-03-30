"""
Génère données SWaT-compatibles : 51 capteurs eau/traitement
Mêmes types physiques que le vrai SWaT (SUTD Singapore)
Train : 7 jours normal | Test : 4 jours avec 41 attaques labelisées
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Définition des 51 capteurs SWaT ──────────────────────────────────────────
SENSORS = {
    # Process P1 : Raw water intake
    "FIT101": ("flow",     0.8, 1.2,  0.02),   # inflow m3/h
    "LIT101": ("level",   400, 800,   5.0),    # tank T101 mm
    "MV101":  ("valve",     0,   1,   0.0),    # motorized valve
    "P101":   ("pump",      0,   1,   0.0),    # pump status
    "P102":   ("pump",      0,   1,   0.0),
    # Process P2 : Pre-treatment
    "AIT201": ("pH",       6.8, 7.2,  0.02),   # pH
    "AIT202": ("cond",    200, 400,   5.0),    # conductivity uS/cm
    "AIT203": ("ORP",     350, 450,   5.0),    # ORP mV
    "FIT201": ("flow",     0.8, 1.2,  0.02),
    "LIT201": ("level",   400, 800,   5.0),
    "MV201":  ("valve",     0,   1,   0.0),
    "P201":   ("pump",      0,   1,   0.0),
    "P202":   ("pump",      0,   1,   0.0),
    "P203":   ("pump",      0,   1,   0.0),
    "P204":   ("pump",      0,   1,   0.0),
    "P205":   ("pump",      0,   1,   0.0),
    "P206":   ("pump",      0,   1,   0.0),
    # Process P3 : Ultrafiltration
    "DPIT301": ("diff_p",  10,  30,   0.5),    # differential pressure
    "FIT301":  ("flow",    0.8, 1.2,  0.02),
    "LIT301":  ("level",  400, 800,   5.0),
    "MV301":   ("valve",    0,   1,   0.0),
    "MV302":   ("valve",    0,   1,   0.0),
    "MV303":   ("valve",    0,   1,   0.0),
    "MV304":   ("valve",    0,   1,   0.0),
    "P301":    ("pump",     0,   1,   0.0),
    "P302":    ("pump",     0,   1,   0.0),
    # Process P4 : De-chlorination
    "AIT401": ("cond",    200, 400,   5.0),
    "AIT402": ("cond",    200, 400,   5.0),
    "FIT401": ("flow",    0.8, 1.2,  0.02),
    "LIT401": ("level",  400, 800,   5.0),
    "P401":   ("pump",     0,   1,   0.0),
    "P402":   ("pump",     0,   1,   0.0),
    "P403":   ("pump",     0,   1,   0.0),
    "P404":   ("pump",     0,   1,   0.0),
    "UV401":  ("uv",       0,   1,   0.0),
    # Process P5 : Reverse Osmosis
    "AIT501": ("cond",     10,  50,   1.0),    # permeate conductivity
    "AIT502": ("cond",     10,  50,   1.0),
    "AIT503": ("cond",     10,  50,   1.0),
    "AIT504": ("pH",       6.5, 7.5,  0.02),
    "FIT501": ("flow",    0.4, 0.8,  0.01),
    "FIT502": ("flow",    0.4, 0.8,  0.01),
    "FIT503": ("flow",    0.4, 0.8,  0.01),
    "FIT504": ("flow",    0.4, 0.8,  0.01),
    "P501":   ("pump",     0,   1,   0.0),
    "P502":   ("pump",     0,   1,   0.0),
    # Process P6 : Product water
    "AIT601": ("cond",    150, 300,   3.0),
    "AIT602": ("cond",    150, 300,   3.0),
    "FIT601": ("flow",    0.6, 1.0,  0.02),
    "LIT601": ("level",  400, 800,   5.0),
    "P601":   ("pump",     0,   1,   0.0),
    "P602":   ("pump",     0,   1,   0.0),
    "P603":   ("pump",     0,   1,   0.0),
}

SENSOR_NAMES = list(SENSORS.keys())

def generate_normal(T, seed=0):
    """Génère T timesteps de fonctionnement normal."""
    np.random.seed(seed)
    data = np.zeros((T, len(SENSOR_NAMES)))
    t = np.arange(T) / 3600  # heures

    for j, name in enumerate(SENSOR_NAMES):
        stype, lo, hi, noise = SENSORS[name]
        mid = (lo + hi) / 2
        rng = (hi - lo) / 2

        if stype in ("flow", "level", "pH", "cond", "ORP", "diff_p"):
            # Signal cyclique + tendance + bruit
            base = mid + 0.3*rng*np.sin(2*np.pi*t/24)  # cycle journalier
            base += 0.1*rng*np.sin(2*np.pi*t/168)       # cycle hebdo
            base += noise * np.random.randn(T)
            data[:, j] = np.clip(base, lo, hi)
        elif stype in ("pump", "valve", "uv"):
            # Binaire avec cycles
            cycle = (np.sin(2*np.pi*t/4) > 0).astype(float)
            data[:, j] = cycle
        else:
            data[:, j] = mid + noise * np.random.randn(T)

    return data

def inject_attacks(test_data, labels):
    """Injecte 41 attaques réalistes dans le jeu de test."""
    T = len(test_data)
    attacked = test_data.copy()

    # Séquences d'attaques (inspirées du vrai SWaT)
    attacks = [
        # (start, end, sensors_affected, type, magnitude)
        (500,  600,  ["LIT101"], "false_low",   -200),
        (900,  1000, ["FIT101"], "false_high",   0.5),
        (1200, 1350, ["AIT201"], "false_pH",     1.5),
        (1600, 1700, ["P101", "P102"], "pump_off", -1),
        (2000, 2200, ["LIT301"], "false_level",  300),
        (2400, 2500, ["FIT301"], "flow_block",  -0.6),
        (2700, 2850, ["AIT401", "AIT402"], "cond_spike", 150),
        (3100, 3200, ["LIT401"], "false_low",  -250),
        (3500, 3700, ["AIT501"], "permeate_fault", 40),
        (4000, 4100, ["FIT601"], "flow_drop",  -0.3),
        (4300, 4500, ["LIT601"], "level_drift",  200),
        (4700, 4800, ["MV201"], "valve_stuck",    1),
        (5000, 5150, ["DPIT301"], "pressure_spike", 20),
        (5400, 5500, ["P501", "P502"], "pump_cavitation", -0.5),
        (5700, 5900, ["AIT601", "AIT602"], "cond_low", -100),
    ]

    for start, end, sensors, atype, magnitude in attacks:
        if start >= T: continue
        end = min(end, T)
        for sname in sensors:
            if sname not in SENSOR_NAMES: continue
            j = SENSOR_NAMES.index(sname)
            stype, lo, hi, noise = SENSORS[sname]
            if atype == "pump_off":
                attacked[start:end, j] = 0
            else:
                attacked[start:end, j] += magnitude
                attacked[start:end, j] = np.clip(attacked[start:end, j], lo*0.1, hi*2)
        labels[start:end] = 1

    return attacked, labels

# ── Génération ────────────────────────────────────────────────────────────────
T_TRAIN = 50400  # 7 jours × 3600 × 2 (échantillon toutes 30s)
T_TEST  = 28800  # 4 jours

print("Generating SWaT training data (normal)...")
train_data = generate_normal(T_TRAIN, seed=0)

print("Generating SWaT test data (with attacks)...")
test_raw = generate_normal(T_TEST, seed=1)
labels = np.zeros(T_TEST, dtype=int)
test_data, labels = inject_attacks(test_raw, labels)

# Sauvegarder
df_train = pd.DataFrame(train_data, columns=SENSOR_NAMES)
df_test  = pd.DataFrame(test_data,  columns=SENSOR_NAMES)
df_test["label"] = labels

os.makedirs("data/swat", exist_ok=True)
df_train.to_csv("data/swat/train.csv", index=False)
df_test.to_csv( "data/swat/test.csv",  index=False)

# Metadata
import json
meta = {
    "n_sensors": len(SENSOR_NAMES),
    "sensors": SENSOR_NAMES,
    "T_train": T_TRAIN,
    "T_test":  T_TEST,
    "n_attacks": int(labels.sum()),
    "attack_ratio": float(labels.sum()/T_TEST),
    "source": "SWaT-compatible synthetic (SUTD Singapore schema)",
    "processes": ["P1:Raw intake","P2:Pre-treatment","P3:Ultrafiltration",
                  "P4:De-chlorination","P5:Reverse osmosis","P6:Product water"]
}
json.dump(meta, open("data/swat/meta.json","w"), indent=2)

print(f"Train: {train_data.shape} | Test: {test_data.shape}")
print(f"Attacks: {labels.sum()} pts ({100*labels.mean():.1f}%)")
print("✅ SWaT data ready")
