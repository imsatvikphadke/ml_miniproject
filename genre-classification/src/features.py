"""Step 1: audio preprocessing + MFCC extraction (LibROSA), plus the fixed 80/20 split."""
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

from . import config as C


def mfcc_from_signal(y: np.ndarray) -> np.ndarray:
    """Waveform (already at C.SR) -> MFCC of shape (N_FRAMES, N_MFCC)."""
    target = C.SR * C.DURATION
    y = np.pad(y, (0, max(0, target - len(y))))[:target]       # consistent length
    mfcc = librosa.feature.mfcc(y=y, sr=C.SR, n_mfcc=C.N_MFCC, hop_length=C.HOP_LENGTH)
    mfcc = mfcc[:, : C.N_FRAMES]                                # 1292 -> 1290 frames
    if mfcc.shape[1] < C.N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, C.N_FRAMES - mfcc.shape[1])))
    return mfcc.T.astype(np.float32)


def load_mfcc(path) -> np.ndarray:
    """Return MFCC as (N_FRAMES, N_MFCC): time on axis 0, coefficients as channels."""
    y, _ = librosa.load(path, sr=C.SR, duration=C.DURATION)
    return mfcc_from_signal(y)


def build_dataset(data_dir=C.DATA_DIR, out=C.CACHE_DIR / "mfcc.npz"):
    data_dir = C.Path(data_dir)
    X, y, files = [], [], []
    for label, genre in enumerate(C.GENRES):
        paths = sorted((data_dir / genre).glob("*.wav"))
        if not paths:
            raise FileNotFoundError(f"No .wav files in {data_dir / genre}")
        print(f"{genre}: {len(paths)} tracks")
        for p in paths:
            X.append(load_mfcc(p)); y.append(label); files.append(p.name)
    X, y, files = np.stack(X), np.array(y), np.array(files)

    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X_train=X[idx_tr], y_train=y[idx_tr], X_test=X[idx_te], y_test=y[idx_te],
        files_train=files[idx_tr], files_test=files[idx_te],
    )
    print(f"Saved {out}: train {len(idx_tr)}, test {len(idx_te)}, shape per track {X.shape[1:]}")


def load_dataset(path=C.CACHE_DIR / "mfcc.npz"):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]
