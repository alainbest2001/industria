"""
models/infraguard.py — InfraGuard AI
LSTM Autoencoder sur données SWaT (51 capteurs eau/traitement)
GPU-accelerated : détecte CUDA automatiquement, fallback CPU transparent.
Modèle entraîné sur GPU, inférence CPU-compatible pour Streamlit Cloud.
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


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=0.0)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(self, hidden, output_dim, n_layers=1, window=96):
        super().__init__()
        self.window = window
        self.lstm   = nn.LSTM(hidden, hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden, output_dim)
    def forward(self, h, c):
        z = h[-1].unsqueeze(1).repeat(1, self.window, 1)
        out, _ = self.lstm(z, (h, c))
        return self.linear(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden=48, n_layers=1, window=96):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden, n_layers)
        self.decoder = LSTMDecoder(hidden, input_dim, n_layers, window)
    def forward(self, x):
        h, c = self.encoder(x)
        return self.decoder(h, c)


class InfraGuardDetector:
    """
    Pipeline InfraGuard — GPU-accelerated.
    Entraîne sur GPU si disponible, scoring en batches optimisés.
    Modèle sauvegardable en CPU pour déploiement Streamlit Cloud.
    """

    def __init__(self, window=96, hidden=64, epochs=25,
                 threshold_pct=95.0, batch_size=256, lr=1e-3, step=4):
        self.window        = window
        self.hidden        = hidden
        self.epochs        = epochs
        self.threshold_pct = threshold_pct
        self.batch_size    = batch_size
        self.lr            = lr
        self.step          = step
        self.scaler        = MinMaxScaler((-1, 1))
        self.model         = None
        self.threshold     = None
        self._losses       = []
        self._fitted       = False
        self.device        = DEVICE

    def _windows(self, data):
        T, D = data.shape
        idx  = np.arange(0, T - self.window + 1, self.step)
        return np.stack([data[i:i+self.window] for i in idx])

    def fit(self, train: np.ndarray, progress_cb=None):
        norm   = self.scaler.fit_transform(train)
        W      = self._windows(norm)
        D      = norm.shape[1]
        Xt     = torch.tensor(W, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, Xt),
                            batch_size=self.batch_size, shuffle=True,
                            pin_memory=(self.device.type == "cuda"),
                            num_workers=0)

        self.model = LSTMAutoencoder(D, self.hidden, window=self.window).to(self.device)
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

        # Calibration seuil sur train
        train_scores = self._score(norm)
        self.threshold = float(np.percentile(train_scores, self.threshold_pct))
        self._fitted = True
        return self

    def _score(self, norm: np.ndarray) -> np.ndarray:
        """Score MSE par timestep sur GPU, résultat rapatrié en CPU numpy."""
        T, D   = norm.shape
        W_arr  = self._windows(norm)
        Xt     = torch.tensor(W_arr, dtype=torch.float32)
        errors = np.zeros((len(Xt), self.window))

        # Batch size plus grand possible sur GPU
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
            s = i * self.step
            e = s + self.window
            score[s:e] += err_row
            count[s:e] += 1
        raw = score / np.maximum(count, 1)
        mn, mx = raw.min(), raw.max()
        return (raw - mn) / (mx - mn + 1e-8)

    def predict(self, test: np.ndarray) -> dict:
        assert self._fitted
        norm        = self.scaler.transform(test)
        scores      = self._score(norm)
        predictions = (scores >= self.threshold).astype(int)
        return {
            "scores":      scores,
            "predictions": predictions,
            "threshold":   self.threshold,
            "losses":      self._losses,
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