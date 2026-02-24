import os, re, math, json, time, random, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- optional audio deps ----------
LIBROSA_OK = False
SOUNDFILE_OK = False
SCIPY_OK = False

try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

try:
    import soundfile as sf
    SOUNDFILE_OK = True
except Exception:
    SOUNDFILE_OK = False

try:
    from scipy.signal import resample_poly
    from scipy.io import wavfile
    from scipy import signal as scipy_signal
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------- ML/DL deps ----------
TORCH_OK = False
SKLEARN_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.manifold import TSNE
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, to_rgb


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA_PATH = r"D:\Other words data\ANIMALS"   # <-- CHANGE THIS
OUTPUT_SUBFOLDER = "results_fair_encoderplus_claf"

# Spectrogram
TARGET_SR = 16000
CLIP_SECONDS = 3.0
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
POWER = 2.0
USE_DB = True

# Training
EPOCHS = 100
BATCH_SIZE = 34
LR_SINGLE = 1e-3
LR_FUSION = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE_LR = 6
GRAD_CLIP_NORM = 5.0

# Split
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.15
USE_GROUP_SPLIT_IF_POSSIBLE = True
RANDOM_SEED = 42

# EncoderPlus (shared by baseline and fusion!)
ENC_CHANNELS = (64, 128, 192)   # 3 conv blocks
ENC_KERNELS  = (7, 5, 3)
ENC_DROPOUT  = 0.30
ENC_LSTM_HIDDEN = 128
ENC_LSTM_LAYERS = 2

EMB_DIM = 128

# Fusion
FUSION_HEADS = 8
FUSION_ATTN_DROPOUT = 0.20
FUSION_MLP_DROPOUT = 0.30

# Plots
DPI = 300
NORMALIZE_CM = True


# =============================================================================
# GLOBAL PLOT STYLE
# =============================================================================
def set_global_plot_style():
    matplotlib.rcParams["font.family"] = "Calibri"
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["axes.labelweight"] = "bold"
    matplotlib.rcParams["axes.titlesize"] = 18
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["xtick.labelsize"] = 11
    matplotlib.rcParams["ytick.labelsize"] = 11
    matplotlib.rcParams["figure.dpi"] = DPI
    matplotlib.rcParams["savefig.dpi"] = DPI

set_global_plot_style()


# =============================================================================
# SEED
# =============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(RANDOM_SEED)


# =============================================================================
# COLOR HELPERS (3 CM themes)
# =============================================================================
PALETTE_16 = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
    "#EE7733", "#0077BB", "#33BBEE", "#EE3377",
    "#CC3311", "#009988", "#AA4499", "#BBBBBB"
]

def _lighten_hex(hex_color: str, amount: float = 0.86) -> str:
    r, g, b = to_rgb(hex_color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return matplotlib.colors.to_hex((r, g, b))

def _make_light_cmap(base_hex: str, lighten: float = 0.86, name="custom"):
    light_hex = _lighten_hex(base_hex, amount=lighten)
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", light_hex, base_hex], N=256)

CM_THEMES_3 = [
    ("Purple",      _make_light_cmap("#6A0DAD", lighten=0.86, name="cm_purple")),
    ("ForestGreen", _make_light_cmap("#228B22", lighten=0.86, name="cm_green")),
    ("Brown",       _make_light_cmap("#7B4A12", lighten=0.86, name="cm_brown")),
    ("SkyBlue",     _make_light_cmap("#56B4E9", lighten=0.86, name="cm_skyblue")),
("darkBlue",     _make_light_cmap("#56B4E9", lighten=0.86, name="cm_darkblue")),
    ("LightPink",   _make_light_cmap("#F4A3C4", lighten=0.88, name="cm_lightpink")),
    ("LightYellow", _make_light_cmap("#F6E58D", lighten=0.88, name="cm_lightyellow")),
    ("Orange",      _make_light_cmap("#F39C12", lighten=0.86, name="cm_orange")),
    ("Gray",        _make_light_cmap("#6E6E6E", lighten=0.90, name="cm_gray")),
]


# =============================================================================
# FILE / PAIRING HELPERS
# =============================================================================
def parse_pair_id(stem: str):
    s = stem.lower().replace(" ", "")
    m = re.search(r"(audio|mic|microphone|piezo)\s*[-_]?(\d+)$", s)
    if not m:
        return None
    return int(m.group(2))

def parse_group_id(audio_path: str) -> str:
    """
    Try to infer a 'group' (speaker/session) from filename/folder.
    If it returns mostly P0, group split cannot work.
    """
    p = Path(audio_path)
    stem = p.stem

    m = re.search(r"(p|s|spk|subj|user)\s*[-_]?(\d+)", stem, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"

    for parent in [p.parent, p.parent.parent]:
        if parent is None:
            continue
        mm = re.search(r"(p|s|spk|subj|user)\s*[-_]?(\d+)", parent.name, flags=re.IGNORECASE)
        if mm:
            return f"{mm.group(1).upper()}{int(mm.group(2))}"

    low = stem.lower()
    for key in ["audio", "mic", "microphone", "piezo"]:
        idx = low.find(key)
        if idx > 0:
            prefix = stem[:idx].rstrip("_- ")
            if len(prefix) >= 2:
                return f"G_{prefix}"

    return "P0"

def scan_paired_dataset(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {root}")

    ignore_folders = {".idea", ".venv", ".venv311", "__pycache__", ".git", OUTPUT_SUBFOLDER}
    class_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in ignore_folders]
    class_dirs = sorted(class_dirs)

    samples = []
    for cdir in class_dirs:
        label = cdir.name
        wavs = sorted(list(cdir.rglob("*.wav")))

        audio_map, piezo_map = {}, {}

        for w in wavs:
            pid = parse_pair_id(w.stem)
            if pid is None:
                continue
            low = w.stem.lower()
            if low.startswith(("audio", "mic", "microphone")):
                audio_map[pid] = str(w)
            elif low.startswith("piezo"):
                piezo_map[pid] = str(w)

        common_ids = sorted(set(audio_map.keys()) & set(piezo_map.keys()))
        for pid in common_ids:
            ap = audio_map[pid]
            pp = piezo_map[pid]
            grp = parse_group_id(ap)
            samples.append({
                "label": label,
                "audio_path": ap,
                "piezo_path": pp,
                "group": grp,
                "pair_id": pid
            })

        print(f"[{label}] paired samples found: {len(common_ids)}")

    if len(samples) == 0:
        raise RuntimeError("No paired (audio_<id> + piezo_<id>) WAV samples were found.")
    return samples


# =============================================================================
# AUDIO LOADING + LOG-MEL EXTRACTION
# =============================================================================
def _safe_load_wav(path: str, target_sr: int):
    if LIBROSA_OK:
        y, _sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), target_sr

    if SOUNDFILE_OK:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != target_sr:
            if SCIPY_OK:
                g = math.gcd(sr, target_sr)
                y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
            else:
                raise RuntimeError("Need scipy to resample when librosa is not installed.")
        return y, target_sr

    if SCIPY_OK:
        sr, y = wavfile.read(path)
        if y.ndim > 1:
            y = y[:, 0]
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = y.astype(np.float32)
        if sr != target_sr:
            g = math.gcd(sr, target_sr)
            y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        return y, target_sr

    raise RuntimeError("Install librosa OR soundfile OR scipy to load WAV files.")

def _fix_length(y: np.ndarray, n_samples: int):
    if len(y) < n_samples:
        return np.pad(y, (0, n_samples - len(y)), mode="constant")
    return y[:n_samples]

def compute_logmel_sequence(path: str):
    y, sr = _safe_load_wav(path, TARGET_SR)
    n = int(TARGET_SR * CLIP_SECONDS)
    y = _fix_length(y, n)

    if LIBROSA_OK:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, power=POWER
        )
        if USE_DB:
            S = librosa.power_to_db(S + 1e-10)
        return S.T.astype(np.float32)

    if SCIPY_OK:
        f, t, Z = scipy_signal.stft(y, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
        mag = np.abs(Z) ** 2
        logmag = 10.0 * np.log10(mag + 1e-10)
        fb = logmag.shape[0]
        if fb >= N_MELS:
            idx = np.linspace(0, fb - 1, N_MELS).astype(int)
            pooled = logmag[idx, :]
        else:
            pooled = np.pad(logmag, ((0, N_MELS - fb), (0, 0)), mode="edge")
        return pooled.T.astype(np.float32)

    raise RuntimeError("Need librosa or scipy for spectrogram feature extraction.")

def per_sample_normalize(X: np.ndarray):
    mu = float(np.mean(X))
    sd = float(np.std(X) + 1e-6)
    return (X - mu) / sd


# =============================================================================
# DATASETS
# =============================================================================
class SpecDatasetSingle(Dataset):
    def __init__(self, X_seq, y):
        self.X = X_seq
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = per_sample_normalize(self.X[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class SpecDatasetFusion(Dataset):
    def __init__(self, X_piezo, X_audio, y):
        self.Xp = X_piezo
        self.Xa = X_audio
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xp = per_sample_normalize(self.Xp[idx])
        xa = per_sample_normalize(self.Xa[idx])
        return (
            torch.tensor(xp, dtype=torch.float32),
            torch.tensor(xa, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# =============================================================================
# SPLITTING
# =============================================================================
def split_indices(y, groups=None, test_size=0.2, seed=42, use_group=True):
    idx = np.arange(len(y))
    if use_group and groups is not None:
        uniq = sorted(set(groups))
        if len(uniq) > 1:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            tr, te = next(gss.split(idx, y, groups))
            return tr, te
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                              stratify=y if len(set(y)) > 1 else None)
    return tr, te


# =============================================================================
# BUILD FEATURE ARRAYS (with caching)
# =============================================================================
def build_feature_arrays(samples, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index_file = cache_dir / "cache_index.json"

    cache_index = {}
    if cache_index_file.exists():
        try:
            cache_index = json.loads(cache_index_file.read_text(encoding="utf-8"))
        except Exception:
            cache_index = {}

    def get_cached_spec(wav_path: str, kind: str):
        key = f"{kind}:{wav_path}"
        if key in cache_index:
            f = cache_dir / cache_index[key]
            if f.exists():
                return np.load(f)
        X = compute_logmel_sequence(wav_path)
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", Path(wav_path).stem)[:80]
        out_name = f"{kind}_{safe_name}_{abs(hash(wav_path)) % (10**10)}.npy"
        out_path = cache_dir / out_name
        np.save(out_path, X)
        cache_index[key] = out_name
        return X

    Xp_list, Xa_list, labels, groups = [], [], [], []
    t0 = time.time()

    for i, s in enumerate(samples, start=1):
        xp = get_cached_spec(s["piezo_path"], "piezo")
        xa = get_cached_spec(s["audio_path"], "audio")

        T = min(xp.shape[0], xa.shape[0])
        Xp_list.append(xp[:T, :])
        Xa_list.append(xa[:T, :])
        labels.append(s["label"])
        groups.append(s["group"])

        if i % 50 == 0:
            print(f"Cached/loaded {i}/{len(samples)} samples...")

    try:
        cache_index_file.write_text(json.dumps(cache_index, indent=2), encoding="utf-8")
    except Exception:
        pass

    X_piezo = np.stack(Xp_list, axis=0).astype(np.float32)
    X_audio = np.stack(Xa_list, axis=0).astype(np.float32)

    print(f"\nBuilt arrays: Piezo {X_piezo.shape} | Audio {X_audio.shape} | Time {time.time()-t0:.1f}s")
    return X_piezo, X_audio, labels, groups


# =============================================================================
# MODELS (shared EncoderPlus)
# =============================================================================
class AttnPool1D(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, out_seq):
        w = torch.softmax(self.scorer(out_seq), dim=1)     # (B,T,1)
        return (w * out_seq).sum(dim=1)                    # (B,H)

class CNNLSTMEncoderPlus(nn.Module):
    def __init__(self, n_features, emb_dim=EMB_DIM):
        super().__init__()
        ch = ENC_CHANNELS
        ks = ENC_KERNELS
        d = ENC_DROPOUT

        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(n_features, ch[0], kernel_size=ks[0], padding=ks[0]//2)
        self.bn1 = nn.BatchNorm1d(ch[0])
        self.drop1 = nn.Dropout(d)

        self.conv2 = nn.Conv1d(ch[0], ch[1], kernel_size=ks[1], padding=ks[1]//2)
        self.bn2 = nn.BatchNorm1d(ch[1])
        self.drop2 = nn.Dropout(d)

        self.conv3 = nn.Conv1d(ch[1], ch[2], kernel_size=ks[2], padding=ks[2]//2)
        self.bn3 = nn.BatchNorm1d(ch[2])
        self.drop3 = nn.Dropout(d)

        self.lstm = nn.LSTM(
            input_size=ch[2],
            hidden_size=ENC_LSTM_HIDDEN,
            num_layers=ENC_LSTM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=d if ENC_LSTM_LAYERS > 1 else 0.0
        )

        self.attnpool = AttnPool1D(ENC_LSTM_HIDDEN)

        self.proj = nn.Sequential(
            nn.LayerNorm(ENC_LSTM_HIDDEN),
            nn.Dropout(d),
            nn.Linear(ENC_LSTM_HIDDEN, emb_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B,F,T)
        x = self.pool(self.act(self.bn1(self.conv1(x)))); x = self.drop1(x)
        x = self.pool(self.act(self.bn2(self.conv2(x)))); x = self.drop2(x)
        x = self.pool(self.act(self.bn3(self.conv3(x)))); x = self.drop3(x)

        x = x.transpose(1, 2)  # (B,T',C)
        out_seq, _ = self.lstm(x)
        pooled = self.attnpool(out_seq)
        return self.proj(pooled)  # (B,EMB_DIM)

class CNNLSTMClassifierPlus(nn.Module):
    """
    Baseline classifier using the SAME EncoderPlus as fusion.
    """
    def __init__(self, n_features, num_classes):
        super().__init__()
        self.enc = CNNLSTMEncoderPlus(n_features=n_features, emb_dim=EMB_DIM)
        self.head = nn.Sequential(
            nn.Dropout(ENC_DROPOUT),
            nn.Linear(EMB_DIM, 128),
            nn.GELU(),
            nn.Dropout(ENC_DROPOUT),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(ENC_DROPOUT),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        e = self.enc(x)
        return self.head(e)

class CLAF_AttentionFusion(nn.Module):
    """
    Fusion model using the SAME EncoderPlus for piezo & audio.
    """
    def __init__(self, piezo_features, audio_features, num_classes):
        super().__init__()
        if EMB_DIM % FUSION_HEADS != 0:
            raise ValueError("EMB_DIM must be divisible by FUSION_HEADS")

        self.piezo_enc = CNNLSTMEncoderPlus(piezo_features, emb_dim=EMB_DIM)
        self.audio_enc = CNNLSTMEncoderPlus(audio_features, emb_dim=EMB_DIM)

        self.attn = nn.MultiheadAttention(
            embed_dim=EMB_DIM,
            num_heads=FUSION_HEADS,
            dropout=FUSION_ATTN_DROPOUT,
            batch_first=True
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(EMB_DIM * 2),
            nn.Dropout(FUSION_MLP_DROPOUT),
            nn.Linear(EMB_DIM * 2, 128),
            nn.GELU(),
            nn.Dropout(FUSION_MLP_DROPOUT),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, piezo_x, audio_x, return_before_after=False):
        p = self.piezo_enc(piezo_x)                 # (B,128)
        a = self.audio_enc(audio_x)                 # (B,128)

        tokens = torch.stack([p, a], dim=1)         # (B,2,128)
        attn_out, attn_w = self.attn(tokens, tokens, tokens)

        flat = attn_out.reshape(attn_out.size(0), -1)  # (B,256)
        z = self.fuse(flat)                            # (B,64)
        logits = self.out(z)

        if return_before_after:
            before = torch.cat([p, a], dim=1)          # (B,256)
            after = z                                   # (B,64)
            return logits, attn_w, before, after

        return logits, attn_w


# =============================================================================
# TRAINER (train/val/test history per epoch)
# =============================================================================
class ModelTrainer:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = []
        self.best_state = None
        self.best_val_acc = 0.0

    def _forward_batch(self, batch, criterion):
        if len(batch) == 2:
            X, y = batch
            X = X.to(self.device); y = y.to(self.device)
            logits = self.model(X)
        else:
            Xp, Xa, y = batch
            Xp = Xp.to(self.device); Xa = Xa.to(self.device); y = y.to(self.device)
            logits, _ = self.model(Xp, Xa)
        loss = criterion(logits, y)
        return loss, logits, y

    def _eval_loader(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                loss, logits, y = self._forward_batch(batch, criterion)
                total_loss += float(loss.item())
                pred = logits.argmax(dim=1)
                total += int(y.size(0))
                correct += int((pred == y).sum().item())
        return total_loss / max(1, len(loader)), correct / max(1, total)

    def train(self, train_loader, val_loader, test_loader, epochs, lr):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=PATIENCE_LR
        )

        self.history = []
        self.best_val_acc = 0.0
        self.best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            tr_loss = 0.0
            tr_correct = 0
            tr_total = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, logits, y = self._forward_batch(batch, criterion)
                loss.backward()

                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

                tr_loss += float(loss.item())
                pred = logits.argmax(dim=1)
                tr_total += int(y.size(0))
                tr_correct += int((pred == y).sum().item())

            tr_loss = tr_loss / max(1, len(train_loader))
            tr_acc = tr_correct / max(1, tr_total)

            va_loss, va_acc = self._eval_loader(val_loader, criterion)
            te_loss, te_acc = self._eval_loader(test_loader, criterion)

            scheduler.step(va_loss)

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            self.history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] TR {tr_acc:.4f} | VA {va_acc:.4f} | TE {te_acc:.4f}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.best_val_acc

    def predict(self, loader):
        self.model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    X, y = batch
                    X = X.to(self.device)
                    logits = self.model(X)
                else:
                    Xp, Xa, y = batch
                    Xp = Xp.to(self.device); Xa = Xa.to(self.device)
                    logits, _ = self.model(Xp, Xa)
                pred = logits.argmax(dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(y.numpy())
        return np.concatenate(all_pred), np.concatenate(all_true)

    def history_df(self):
        return pd.DataFrame(self.history)


# =============================================================================
# EXPORTS
# =============================================================================
# =============================================================================
# =============================================================================
# EXPORTS
# =============================================================================
def save_history_csv(trainer: ModelTrainer, out_csv: Path, meta: dict):
    df = trainer.history_df()
    for k, v in meta.items():
        df[k] = v
    df.to_csv(out_csv, index=False)
    return df

def plot_history(trainer: ModelTrainer, out_png: Path, title: str):
    df = trainer.history_df()
    if df.empty:
        return
    epochs = df["epoch"].values

    fig, ax1 = plt.subplots(figsize=(11.0, 5.2))
    ax1.plot(epochs, df["train_acc"]*100, linewidth=2.6, label="Train Acc (%)")
    ax1.plot(epochs, df["val_acc"]*100, linewidth=2.6, linestyle="--", label="Val Acc (%)")
    ax1.plot(epochs, df["test_acc"]*100, linewidth=2.6, linestyle=":", label="Test Acc (%)")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.18)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(epochs, df["train_loss"], linewidth=2.2, alpha=0.9, label="Train Loss")
    ax2.plot(epochs, df["val_loss"], linewidth=2.2, linestyle="--", alpha=0.9, label="Val Loss")
    ax2.plot(epochs, df["test_loss"], linewidth=2.2, linestyle=":", alpha=0.9, label="Test Loss")
    ax2.set_ylabel("Loss", fontweight="bold")
    ax2.spines["top"].set_visible(False)

    ax1.set_title(title, fontweight="bold", pad=10)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, loc="center right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def save_confusion_matrix_csv(cm, classes, out_csv: Path, normalize=True):
    cm = cm.astype(float)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_csv, index=True)

def plot_confusion_matrix(cm, classes, out_file: Path, title: str, cmap, normalize=True):
    cm = cm.astype(float)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1 if normalize else None)

    ax.set_title(title, fontweight="bold", fontsize=26, pad=10)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(classes, fontweight="bold")

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n):
        for c in range(n):
            ax.text(c, r, f"{cm[r,c]:.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def save_confusion_matrix_3colors(cm, classes, out_prefix: Path, title_prefix: str):
    for theme_name, cmap in CM_THEMES_3:
        out_png = out_prefix.parent / f"{out_prefix.name}__{theme_name}.png"
        plot_confusion_matrix(cm, classes, out_png, f"{title_prefix} ({theme_name})", cmap, normalize=NORMALIZE_CM)


# =============================================================================
# EMBEDDING EXTRACTORS (NO MODEL CHANGES)
# =============================================================================
def extract_embeddings_single(model, loader, device):
    """
    For Piezo/Audio single-modality models (CNNLSTMClassifierPlus).
    Returns:
      emb: (N, EMB_DIM)
      y:   (N,)
    """
    model.eval()
    embs, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            e = model.enc(X)  # encoder embedding (B, EMB_DIM)
            embs.append(e.detach().cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(embs), np.concatenate(ys)

def extract_embeddings_fusion_before_after(model, loader, device):
    """
    For Fusion model (CLAF_AttentionFusion).
    Returns:
      before: (N, 2*EMB_DIM)  concat [p||a]
      after:  (N, 64)         fused embedding z
      y:      (N,)
    """
    model.eval()
    befores, afters, ys = [], [], []
    with torch.no_grad():
        for Xp, Xa, y in loader:
            Xp = Xp.to(device)
            Xa = Xa.to(device)
            logits, attn_w, before, after = model(Xp, Xa, return_before_after=True)
            befores.append(before.detach().cpu().numpy())
            afters.append(after.detach().cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(befores), np.vstack(afters), np.concatenate(ys)


# =============================================================================
# t-SNE: MULTI-WINDOW (3 windows per sample) to make clusters "fuller"
# (NO effect on training/accuracy; this is only for visualization.)
# =============================================================================

# ---- t-SNE window settings ----
TSNE_WINDOWS_PER_SAMPLE = 3     # <--- 3 windows per file/sample
TSNE_WINDOW_RATIO = 0.70        # each window uses 70% of frames (overlapping)
TSNE_MIN_FRAMES = 12            # safety: don't slice too short (after pooling)
TSNE_DOT_SIZE = 95              # keep your existing style

def _style_axes_like_cm(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=16)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=16)
    ax.tick_params(labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.8)
    ax.grid(True, alpha=0.18)

def _make_window_slices(T: int, n_windows: int, ratio: float):
    """
    Deterministic evenly-spaced windows along time axis.
    Returns a list of slice objects for frames [start : start+winT].
    """
    if T <= TSNE_MIN_FRAMES or n_windows <= 1:
        return [slice(0, T)]

    winT = int(max(TSNE_MIN_FRAMES, round(T * float(ratio))))
    winT = min(winT, T)

    if winT >= T:
        return [slice(0, T)]

    max_start = T - winT
    if n_windows == 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, n_windows).round().astype(int).tolist()

    # ensure unique-ish starts, but keep length
    slices = [slice(s, s + winT) for s in starts]
    return slices

def extract_embeddings_single_multiwindow(model, loader, device,
                                         n_windows=TSNE_WINDOWS_PER_SAMPLE,
                                         ratio=TSNE_WINDOW_RATIO):
    """
    Create multiple embeddings per sample by slicing along time dimension.
    Returns:
      emb: (N*n_windows, EMB_DIM)
      y:   (N*n_windows,)
      sample_id: (N*n_windows,)  index of original sample in this loader order
      window_id: (N*n_windows,)  0..n_windows-1
    """
    model.eval()
    embs, ys, sample_ids, window_ids = [], [], [], []
    global_sample_counter = 0

    with torch.no_grad():
        for X, y in loader:
            # X: (B, T, F)
            B = X.size(0)
            for i in range(B):
                xi = X[i]        # (T, F) on CPU
                yi = int(y[i].item())
                T = int(xi.shape[0])
                slices = _make_window_slices(T, n_windows, ratio)

                for w_id, sl in enumerate(slices):
                    xw = xi[sl, :].unsqueeze(0).to(device)  # (1, Tw, F)
                    e = model.enc(xw)                       # (1, EMB_DIM)
                    embs.append(e.detach().cpu().numpy())
                    ys.append(yi)
                    sample_ids.append(global_sample_counter)
                    window_ids.append(w_id)

                global_sample_counter += 1

    emb = np.vstack(embs)
    y_out = np.array(ys, dtype=int)
    return emb, y_out, np.array(sample_ids, dtype=int), np.array(window_ids, dtype=int)

def extract_embeddings_fusion_before_after_multiwindow(model, loader, device,
                                                       n_windows=TSNE_WINDOWS_PER_SAMPLE,
                                                       ratio=TSNE_WINDOW_RATIO):
    """
    Multi-window version for fusion model.
    Returns:
      before: (N*n_windows, 2*EMB_DIM)
      after:  (N*n_windows, 64)
      y:      (N*n_windows,)
      sample_id, window_id
    """
    model.eval()
    befores, afters, ys, sample_ids, window_ids = [], [], [], [], []
    global_sample_counter = 0

    with torch.no_grad():
        for Xp, Xa, y in loader:
            # Xp/Xa: (B, T, F)
            B = Xp.size(0)
            for i in range(B):
                xpi = Xp[i]  # (T, F) CPU
                xai = Xa[i]
                yi = int(y[i].item())

                Tp = int(xpi.shape[0])
                Ta = int(xai.shape[0])
                T = min(Tp, Ta)
                if T <= 1:
                    global_sample_counter += 1
                    continue

                xpi = xpi[:T, :]
                xai = xai[:T, :]

                slices = _make_window_slices(T, n_windows, ratio)

                for w_id, sl in enumerate(slices):
                    xpw = xpi[sl, :].unsqueeze(0).to(device)  # (1, Tw, F)
                    xaw = xai[sl, :].unsqueeze(0).to(device)

                    logits, attn_w, before, after = model(xpw, xaw, return_before_after=True)
                    befores.append(before.detach().cpu().numpy())
                    afters.append(after.detach().cpu().numpy())
                    ys.append(yi)
                    sample_ids.append(global_sample_counter)
                    window_ids.append(w_id)

                global_sample_counter += 1

    return (np.vstack(befores), np.vstack(afters), np.array(ys, dtype=int),
            np.array(sample_ids, dtype=int), np.array(window_ids, dtype=int))

def plot_tsne_single(emb, y, class_names, out_png: Path, out_csv: Path,
                     title: str, seed=42, sample_id=None, window_id=None):
    """
    One t-SNE figure + CSV for a single embedding set.
    CSV includes optional sample_id/window_id (for multi-window).
    """
    perplexity = min(35, max(5, (len(y) // 20)))
    tsne = TSNE(n_components=2, random_state=seed, init="pca",
                learning_rate="auto", perplexity=perplexity)
    Z = tsne.fit_transform(emb)

    df = pd.DataFrame({"tsne1": Z[:, 0], "tsne2": Z[:, 1], "label_id": y})
    df["label"] = [class_names[i] for i in y]
    if sample_id is not None:
        df["sample_id"] = sample_id
    if window_id is not None:
        df["window_id"] = window_id
    df.to_csv(out_csv, index=False)

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">", "8"]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    for cid in sorted(np.unique(y)):
        mask = (y == cid)
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=TSNE_DOT_SIZE, alpha=0.86,
            marker=markers[cid % len(markers)],
            edgecolors="white", linewidths=0.6,
            c=PALETTE_16[cid % len(PALETTE_16)],
            label=class_names[cid]
        )

    ax.set_title(title, fontweight="bold", fontsize=20, pad=12)
    _style_axes_like_cm(ax, "t-SNE 1", "t-SNE 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_tsne_before_after_separate(before_emb, after_emb, y, class_names,
                                   out_before_png: Path, out_before_csv: Path,
                                   out_after_png: Path, out_after_csv: Path,
                                   seed=42, sample_id=None, window_id=None):
    """
    Two SEPARATE t-SNE figures + CSV:
      - BEFORE fusion (concat [p||a])
      - AFTER fusion (z)
    CSV includes optional sample_id/window_id.
    """
    perplexity = min(35, max(5, (len(y) // 20)))

    tsne1 = TSNE(n_components=2, random_state=seed, init="pca",
                 learning_rate="auto", perplexity=perplexity)
    Z1 = tsne1.fit_transform(before_emb)
    df1 = pd.DataFrame({"tsne1": Z1[:, 0], "tsne2": Z1[:, 1], "label_id": y})
    df1["label"] = [class_names[i] for i in y]
    if sample_id is not None:
        df1["sample_id"] = sample_id
    if window_id is not None:
        df1["window_id"] = window_id
    df1.to_csv(out_before_csv, index=False)

    tsne2 = TSNE(n_components=2, random_state=seed, init="pca",
                 learning_rate="auto", perplexity=perplexity)
    Z2 = tsne2.fit_transform(after_emb)
    df2 = pd.DataFrame({"tsne1": Z2[:, 0], "tsne2": Z2[:, 1], "label_id": y})
    df2["label"] = [class_names[i] for i in y]
    if sample_id is not None:
        df2["sample_id"] = sample_id
    if window_id is not None:
        df2["window_id"] = window_id
    df2.to_csv(out_after_csv, index=False)

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">", "8"]

    def _plot(Z, out_png: Path, title: str):
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        for cid in sorted(np.unique(y)):
            mask = (y == cid)
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                s=TSNE_DOT_SIZE, alpha=0.86,
                marker=markers[cid % len(markers)],
                edgecolors="white", linewidths=0.6,
                c=PALETTE_16[cid % len(PALETTE_16)],
                label=class_names[cid]
            )
        ax.set_title(title, fontweight="bold", fontsize=20, pad=12)
        _style_axes_like_cm(ax, "t-SNE 1", "t-SNE 2")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=11)
        plt.tight_layout()
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    _plot(Z1, out_before_png, "t-SNE BEFORE Fusion (concat [p||a])")
    _plot(Z2, out_after_png,  "t-SNE AFTER Fusion (z)")


# =============================================================================
# BUILD LOADERS
# =============================================================================
def build_loaders_single(X, y, groups):
    tr_idx, te_idx = split_indices(
        y, groups=groups, test_size=TEST_SIZE, seed=RANDOM_SEED,
        use_group=USE_GROUP_SPLIT_IF_POSSIBLE
    )
    y_tr = y[tr_idx]
    g_tr = None if groups is None else np.array(groups)[tr_idx]

    tr2_rel, va_rel = split_indices(
        y_tr, groups=g_tr, test_size=VAL_SIZE_FROM_TRAIN, seed=RANDOM_SEED,
        use_group=USE_GROUP_SPLIT_IF_POSSIBLE
    )

    tr2_idx = tr_idx[tr2_rel]
    va_idx = tr_idx[va_rel]

    dl_tr = DataLoader(SpecDatasetSingle(X[tr2_idx], y[tr2_idx]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(SpecDatasetSingle(X[va_idx],  y[va_idx]),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(SpecDatasetSingle(X[te_idx],  y[te_idx]),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return dl_tr, dl_va, dl_te

def build_loaders_fusion(Xp, Xa, y, groups):
    tr_idx, te_idx = split_indices(
        y, groups=groups, test_size=TEST_SIZE, seed=RANDOM_SEED,
        use_group=USE_GROUP_SPLIT_IF_POSSIBLE
    )
    y_tr = y[tr_idx]
    g_tr = None if groups is None else np.array(groups)[tr_idx]

    tr2_rel, va_rel = split_indices(
        y_tr, groups=g_tr, test_size=VAL_SIZE_FROM_TRAIN, seed=RANDOM_SEED,
        use_group=USE_GROUP_SPLIT_IF_POSSIBLE
    )

    tr2_idx = tr_idx[tr2_rel]
    va_idx = tr_idx[va_rel]

    dl_tr = DataLoader(SpecDatasetFusion(Xp[tr2_idx], Xa[tr2_idx], y[tr2_idx]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(SpecDatasetFusion(Xp[va_idx],  Xa[va_idx],  y[va_idx]),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(SpecDatasetFusion(Xp[te_idx],  Xa[te_idx],  y[te_idx]),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return dl_tr, dl_va, dl_te


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
def run_single(modality_name, X, y, groups, class_names, results_folder: Path, device):
    dl_tr, dl_va, dl_te = build_loaders_single(X, y, groups)

    model = CNNLSTMClassifierPlus(n_features=X.shape[-1], num_classes=len(class_names))
    trainer = ModelTrainer(model, device=device)

    print(f"\n{'='*70}\nTraining {modality_name} - CNN-LSTM (EncoderPlus baseline)\n{'='*70}")
    best_val = trainer.train(dl_tr, dl_va, dl_te, epochs=EPOCHS, lr=LR_SINGLE)

    y_pred, y_true = trainer.predict(dl_te)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    save_history_csv(trainer, results_folder / f"{modality_name}_CNNLSTM_EncoderPlus_history.csv",
                     meta={"Modality": modality_name, "Model": "CNNLSTM_EncoderPlus", "LR": LR_SINGLE})
    plot_history(trainer, results_folder / f"{modality_name}_CNNLSTM_EncoderPlus_history.png",
                 title=f"{modality_name} - CNNLSTM EncoderPlus (Train/Val/Test)")

    save_confusion_matrix_csv(cm, class_names,
                             results_folder / f"{modality_name}_CNNLSTM_EncoderPlus_confusion.csv",
                             normalize=NORMALIZE_CM)
    save_confusion_matrix_3colors(
        cm, class_names,
        out_prefix=results_folder / f"{modality_name}_CNNLSTM_EncoderPlus_confusion",
        title_prefix=f"{modality_name} CNNLSTM EncoderPlus"
    )

    # ---- t-SNE (FULLER): 3 windows per test sample ----
    emb, y_ts, sid, wid = extract_embeddings_single_multiwindow(trainer.model, dl_te, device=device)
    plot_tsne_single(
        emb=emb,
        y=y_ts,
        class_names=class_names,
        out_png=results_folder / f"{modality_name}_tSNE.png",
        out_csv=results_folder / f"{modality_name}_tSNE.csv",
        title=f"t-SNE of {modality_name} Embeddings (Test Set)  [3 windows/sample]",
        seed=RANDOM_SEED,
        sample_id=sid,
        window_id=wid
    )

    return {"Modality": modality_name, "Model": "CNNLSTM_EncoderPlus",
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "BestVal": best_val}


def run_fusion(Xp, Xa, y, groups, class_names, results_folder: Path, device):
    dl_tr, dl_va, dl_te = build_loaders_fusion(Xp, Xa, y, groups)

    model = CLAF_AttentionFusion(piezo_features=Xp.shape[-1], audio_features=Xa.shape[-1], num_classes=len(class_names))
    trainer = ModelTrainer(model, device=device)

    print(f"\n{'='*70}\nTraining Fused - CLAF Attention Fusion (same EncoderPlus)\n{'='*70}")
    best_val = trainer.train(dl_tr, dl_va, dl_te, epochs=EPOCHS, lr=LR_FUSION)

    y_pred, y_true = trainer.predict(dl_te)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    save_history_csv(trainer, results_folder / "Fusion_CLAF_history.csv",
                     meta={"Modality": "Fused", "Model": "CLAF", "LR": LR_FUSION})
    plot_history(trainer, results_folder / "Fusion_CLAF_history.png",
                 title="Fused - CLAF (Train/Val/Test)")

    save_confusion_matrix_csv(cm, class_names,
                             results_folder / "Fusion_CLAF_confusion.csv",
                             normalize=NORMALIZE_CM)
    save_confusion_matrix_3colors(
        cm, class_names,
        out_prefix=results_folder / "Fusion_CLAF_confusion",
        title_prefix="Fusion CLAF"
    )

    # ---- Fusion t-SNE (FULLER): 3 windows per test sample ----
    before_emb, after_emb, ys, sid, wid = extract_embeddings_fusion_before_after_multiwindow(
        trainer.model, dl_te, device=device
    )

    plot_tsne_before_after_separate(
        before_emb=before_emb,
        after_emb=after_emb,
        y=ys,
        class_names=class_names,
        out_before_png=results_folder / "Fusion_tSNE_BEFORE.png",
        out_before_csv=results_folder / "Fusion_tSNE_BEFORE.csv",
        out_after_png=results_folder / "Fusion_tSNE_AFTER.png",
        out_after_csv=results_folder / "Fusion_tSNE_AFTER.csv",
        seed=RANDOM_SEED,
        sample_id=sid,
        window_id=wid
    )

    # optional extra: after-only plot + CSV
    plot_tsne_single(
        emb=after_emb,
        y=ys,
        class_names=class_names,
        out_png=results_folder / "Fusion_tSNE_After_only.png",
        out_csv=results_folder / "Fusion_tSNE_After_only.csv",
        title="t-SNE of Fusion Embeddings (After) (Test Set)  [3 windows/sample]",
        seed=RANDOM_SEED,
        sample_id=sid,
        window_id=wid
    )

    return {"Modality": "Fused", "Model": "CLAF",
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "BestVal": best_val}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 90)
    print("FAIR: Baselines (EncoderPlus) vs CLAF Fusion (same EncoderPlus)")
    print("=" * 90)

    if not TORCH_OK:
        print("PyTorch is required. Install: pip install torch")
        return
    if not SKLEARN_OK:
        print("scikit-learn is required. Install: pip install scikit-learn")
        return
    if not (LIBROSA_OK or SCIPY_OK):
        print("Need librosa (recommended) or scipy for spectrogram extraction.")
        print("Install: pip install librosa soundfile scipy")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"librosa: {LIBROSA_OK} | soundfile: {SOUNDFILE_OK} | scipy: {SCIPY_OK}")

    samples = scan_paired_dataset(DATA_PATH)
    df_samples = pd.DataFrame(samples)
    print("\nSample preview:")
    print(df_samples.head(5).to_string(index=False))

    results_folder = Path(DATA_PATH) / OUTPUT_SUBFOLDER
    results_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nResults folder: {results_folder}")

    cache_dir = results_folder / "spec_cache"
    Xp, Xa, labels_str, groups = build_feature_arrays(samples, cache_dir)

    le = LabelEncoder()
    y = le.fit_transform(labels_str)
    class_names = list(le.classes_)

    print(f"\nClasses ({len(class_names)}): {class_names}")
    print("\nGroup stats (IMPORTANT):")
    print("Unique groups:", len(set(groups)))
    print("Top groups:", Counter(groups).most_common(10))
    if len(set(groups)) <= 1:
        print("\nWARNING: Only ONE group detected -> group split cannot work. "
              "Your split is effectively stratified random.\n"
              "If you want stricter evaluation, encode session IDs in filenames/folders "
              "and update parse_group_id().")

    summaries = []
    summaries.append(run_single("Piezo", Xp, y, groups, class_names, results_folder, device))
    summaries.append(run_single("Audio", Xa, y, groups, class_names, results_folder, device))
    summaries.append(run_fusion(Xp, Xa, y, groups, class_names, results_folder, device))

    pd.DataFrame(summaries).sort_values("Accuracy", ascending=False).to_csv(results_folder / "GLOBAL_SUMMARY.csv", index=False)

    print("\n" + "=" * 90)
    print("DONE. Outputs saved:")
    print("- *_history.csv + *_history.png")
    print("- Confusion PNGs (Purple/Green/Brown) + confusion CSV")
    print("- Piezo_tSNE.png/.csv and Audio_tSNE.png/.csv  [NOW 3 windows per sample]")
    print("- Fusion_tSNE_BEFORE.png/.csv and Fusion_tSNE_AFTER.png/.csv (+ After_only)  [NOW 3 windows per sample]")
    print("- GLOBAL_SUMMARY.csv")
    print("=" * 90)

if __name__ == "__main__":
    main()
