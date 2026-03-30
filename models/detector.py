"""
models/detector.py — SensorGuard AI v3
LSTM Autoencoder Seq2Seq — GPU-accelerated
Détecte CUDA automatiquement, fallback CPU transparent.
Compatible Streamlit Cloud (CPU) et station RTX (GPU).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Architecture LSTM Autoencoder ─────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=0.0)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(self, hidden, output_dim, n_layers=1, window=64):
        super().__init__()
        self.window = window
        self.lstm   = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden, output_dim)
    def forward(self, h, c):
        z = h[-1].unsqueeze(1).repeat(1, self.window, 1)
        out, _ = self.lstm(z, (h, c))
        return self.linear(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden=32, n_layers=1, window=64):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden, n_layers)
        self.decoder = LSTMDecoder(hidden, input_dim, n_layers, window)
    def forward(self, x):
        h, c = self.encoder(x)
        return self.decoder(h, c)


# ── Composants pipeline ───────────────────────────────────────────────────────

class PhysicalNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler((-1, 1))
    def fit(self, train):
        self.scaler.fit(train); return self
    def transform(self, data):
        return self.scaler.transform(data)
    def fit_transform(self, train):
        return self.scaler.fit_transform(train)


def make_windows(data: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    T, D = data.shape
    idx  = np.arange(0, T - window + 1, step)
    return np.stack([data[i:i+window] for i in idx])


class LSTMAnomalyScorer:
    def __init__(self, window=64, hidden=32, n_layers=1, epochs=30,
                 batch_size=64, lr=1e-3, step=1):
        self.window     = window
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.step       = step
        self.model      = None
        self._losses    = []
        self.device     = DEVICE

    def fit(self, train_norm: np.ndarray, progress_cb=None):
        W  = make_windows(train_norm, self.window, self.step)
        D  = train_norm.shape[1]
        Xt = torch.tensor(W, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, Xt),
                            batch_size=self.batch_size, shuffle=True,
                            pin_memory=(self.device.type == "cuda"))

        self.model = LSTMAutoencoder(D, self.hidden, self.n_layers, self.window).to(self.device)
        opt  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.MSELoss()

        self.model.train()
        self._losses = []
        for ep in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                out  = self.model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= len(Xt)
            self._losses.append(ep_loss)
            if progress_cb:
                progress_cb(ep + 1, ep_loss)
        return self

    def score(self, test_norm: np.ndarray) -> np.ndarray:
        T, D   = test_norm.shape
        W_arr  = make_windows(test_norm, self.window, 1)
        Xt     = torch.tensor(W_arr, dtype=torch.float32)
        errors = np.zeros((len(Xt), self.window))

        score_batch = 512 if self.device.type == "cuda" else 256

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xt), score_batch):
                batch = Xt[i:i+score_batch].to(self.device, non_blocking=True)
                rec   = self.model(batch).cpu().numpy()
                orig  = W_arr[i:i+score_batch]
                errors[i:i+score_batch] = ((orig - rec) ** 2).mean(axis=2)

        score = np.zeros(T)
        count = np.zeros(T)
        for i, err_row in enumerate(errors):
            t_start = i
            t_end   = i + self.window
            score[t_start:t_end] += err_row
            count[t_start:t_end] += 1

        count = np.maximum(count, 1)
        raw   = score / count
        mn, mx = raw.min(), raw.max()
        return (raw - mn) / (mx - mn + 1e-8)


class AdaptiveThreshold:
    def __init__(self, percentile=95.0):
        self.percentile = percentile
        self.threshold  = None
    def fit(self, train_scores):
        self.threshold = float(np.percentile(train_scores, self.percentile))
        return self.threshold
    def predict(self, scores):
        assert self.threshold is not None
        return (scores >= self.threshold).astype(int)


# ── Pipeline complet ──────────────────────────────────────────────────────────

class SensorGuardDetector:
    """
    Pipeline SensorGuard AI v3 — GPU-accelerated :
      PhysicalNormalizer → LSTMAutoencoder → AdaptiveThreshold
    Entraîne sur GPU si disponible, scoring rapide en batches.
    """

    def __init__(self, window=64, threshold_pct=94.0, hidden=32, epochs=30):
        self.normalizer    = PhysicalNormalizer()
        self.scorer        = LSTMAnomalyScorer(
            window=window, hidden=hidden, epochs=epochs)
        self.thresholder   = AdaptiveThreshold(percentile=threshold_pct)
        self.threshold_pct = threshold_pct
        self._fitted       = False

    def fit(self, train: np.ndarray, progress_cb=None):
        norm_train   = self.normalizer.fit_transform(train)
        self.scorer.fit(norm_train, progress_cb=progress_cb)
        train_scores = self.scorer.score(norm_train)
        self.thresholder.fit(train_scores)
        self._fitted = True
        return self

    def predict(self, test: np.ndarray) -> dict:
        assert self._fitted, "Call fit() before predict()"
        norm_test   = self.normalizer.transform(test)
        scores      = self.scorer.score(norm_test)
        predictions = self.thresholder.predict(scores)
        return {
            "scores":      scores,
            "predictions": predictions,
            "threshold":   self.thresholder.threshold,
            "losses":      self.scorer._losses,
        }

    def evaluate(self, predictions, labels) -> dict:
        TP = int(np.sum((predictions == 1) & (labels == 1)))
        FP = int(np.sum((predictions == 1) & (labels == 0)))
        TN = int(np.sum((predictions == 0) & (labels == 0)))
        FN = int(np.sum((predictions == 0) & (labels == 1)))
        p  = TP / (TP + FP + 1e-8)
        r  = TP / (TP + FN + 1e-8)
        f1 = 2*p*r / (p + r + 1e-8)
        return {
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "fpr":       round(FP / (FP + TN + 1e-8), 4),
        }