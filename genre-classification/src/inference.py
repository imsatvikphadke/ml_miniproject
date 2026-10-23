"""Loads the trained artifacts once and serves predictions for the dashboard."""
import io
from dataclasses import dataclass, field

import joblib
import librosa
import numpy as np
from sklearn.decomposition import PCA

from . import classical, dcnn, features
from . import config as C

FEATURE_SETS = ["raw_mfcc", "pca_mfcc", "dcnn_layer1", "dcnn_layer2"]
FEATURE_NAMES = {"raw_mfcc": "Raw MFCC (25,800)", "pca_mfcc": "PCA-50",
                 "dcnn_layer1": "DCNN layer 1", "dcnn_layer2": "DCNN layer 2"}
LABELS = ["Classical", "Hip-hop", "Metal", "Pop", "Blues"]


@dataclass
class Bundle:
    model: object
    extractor: object
    pca: PCA
    clfs: dict
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    files_test: np.ndarray
    feats_tr: dict = field(default_factory=dict)
    feats_te: dict = field(default_factory=dict)
    test_pred: dict = field(default_factory=dict)   # (features, clf) -> test predictions
    train_acc: dict = field(default_factory=dict)
    test_acc: dict = field(default_factory=dict)


def featurize(b: Bundle, X: np.ndarray) -> dict:
    flat = X.reshape(len(X), -1)
    out = {"raw_mfcc": flat, "pca_mfcc": b.pca.transform(flat)}
    for i, o in enumerate(b.extractor(X, training=False), 1):
        out[f"dcnn_layer{i}"] = np.asarray(o).reshape(len(X), -1)
    return out


def load_bundle() -> Bundle:
    d = np.load(C.CACHE_DIR / "mfcc.npz")
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    files = d["files_test"] if "files_test" in d else np.array([f"test_{i}" for i in range(len(yte))])
    model = dcnn.load_dcnn()
    b = Bundle(model, dcnn.make_extractor(model), None, {}, Xtr, ytr, Xte, yte, files)

    pkl = C.CACHE_DIR / "classical.joblib"        # delete this file to force a refit
    if pkl.exists():
        saved = joblib.load(pkl); b.pca, b.clfs = saved["pca"], saved["clfs"]
        b.feats_tr, b.feats_te = featurize(b, Xtr), featurize(b, Xte)
    else:
        b.pca = PCA(n_components=min(C.PCA_COMPONENTS, len(Xtr)), random_state=C.SEED)
        b.pca.fit(Xtr.reshape(len(Xtr), -1))
        b.feats_tr, b.feats_te = featurize(b, Xtr), featurize(b, Xte)
        for f in FEATURE_SETS:
            for name, clf in classical.make_classifiers().items():
                b.clfs[(f, name)] = clf.fit(b.feats_tr[f], ytr)
        joblib.dump({"pca": b.pca, "clfs": b.clfs}, pkl)

    for (f, name), clf in b.clfs.items():
        b.test_pred[(f, name)] = clf.predict(b.feats_te[f])
        b.train_acc[(f, name)] = float(clf.score(b.feats_tr[f], ytr))
        b.test_acc[(f, name)] = float((b.test_pred[(f, name)] == yte).mean())
    p_tr = b.model(Xtr, training=False).numpy().argmax(1)
    p_te = b.model(Xte, training=False).numpy()
    b.test_pred[("mfcc_sequence", "DCNN")] = p_te.argmax(1)
    b.train_acc[("mfcc_sequence", "DCNN")] = float((p_tr == ytr).mean())
    b.test_acc[("mfcc_sequence", "DCNN")] = float((p_te.argmax(1) == yte).mean())
    return b


def predict(b: Bundle, X: np.ndarray) -> dict:
    """X: (n, 1290, 20). Returns CNN probabilities and every classical model's label."""
    probs = b.model(X, training=False).numpy()
    feats = featurize(b, X)
    preds = {k: clf.predict(feats[k[0]]) for k, clf in b.clfs.items()}
    return {"cnn_probs": probs, "preds": preds}


def load_audio(data: bytes) -> np.ndarray:
    y, _ = librosa.load(io.BytesIO(data), sr=C.SR, mono=True)
    return y


def mfcc_window(y: np.ndarray, start_sec: float = 0.0) -> np.ndarray:
    s = int(start_sec * C.SR)
    return features.mfcc_from_signal(y[s: s + C.SR * C.DURATION])
