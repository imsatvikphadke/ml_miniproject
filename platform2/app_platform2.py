from __future__ import annotations
import io, json, time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "genres_original"
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
GENRES = ["classical", "hiphop", "metal", "pop", "blues"]
SEED = 42
TEST_SIZE = 0.2
FEATURE_VERSION = "summary_mfcc_chroma_v1"

st.set_page_config(page_title="Music Genre Classifier — Platform 2", page_icon="🎵", layout="wide")
st.title("🎵 Music Genre Classifier — Platform 2")
st.caption("Our proposed approach: KNN, Extra Trees, and Histogram Gradient Boosting. These are additional models, not the four classical algorithms listed in the reference paper.")

@st.cache_data(show_spinner=False)
def extract_features_from_path(path_string: str) -> np.ndarray:
    y, sr = librosa.load(path_string, sr=22050, mono=True, duration=30)
    return extract_features_from_signal(y, sr)

def extract_features_from_signal(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size == 0:
        raise ValueError("The uploaded audio file contains no readable audio samples.")
    if len(y) < sr:
        y = np.pad(y, (0, max(0, sr-len(y))))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    arrays = [mfcc, chroma, centroid, bandwidth, rolloff, zcr, rms]
    feats = []
    for arr in arrays:
        feats.extend(np.mean(arr, axis=1).tolist())
        feats.extend(np.std(arr, axis=1).tolist())
    return np.asarray(feats, dtype=np.float32)

def find_audio_files():
    rows = []
    for genre in GENRES:
        folder = DATA_DIR / genre
        if folder.exists():
            for p in sorted(folder.glob("*")):
                if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                    rows.append((str(p), genre))
    return rows

@st.cache_resource(show_spinner=False)
def train_models(data_signature: tuple):
    files = find_audio_files()
    X, y, paths, errors = [], [], [], []
    start = time.perf_counter()
    progress = st.progress(0, text="Extracting audio features…")
    for i, (path, label) in enumerate(files):
        try:
            X.append(extract_features_from_path(path))
            y.append(label)
            paths.append(path)
        except Exception as e:
            errors.append(f"{Path(path).name}: {e}")
        progress.progress((i+1)/max(1, len(files)), text=f"Extracting features: {i+1}/{len(files)}")
    progress.empty()
    if len(set(y)) < 2 or len(y) < 10:
        raise RuntimeError("Not enough readable audio files. Add audio files to data/genres_original/<genre>/ for at least two genres.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "Hist. Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=SEED),
    }
    results, fitted = [], {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = model.predict(X_test)
        prediction_seconds = time.perf_counter() - t1
        results.append({"Model": name, "Test accuracy": accuracy_score(y_test, pred), "Train time (s)": train_seconds,
                        "Test prediction time (s)": prediction_seconds, "Test samples": len(y_test)})
        fitted[name] = model
    elapsed = time.perf_counter() - start
    return {"models": fitted, "results": pd.DataFrame(results), "X_test": X_test, "y_test": y_test,
            "paths_test": paths_test, "features": X.shape[1], "total_tracks": len(y),
            "counts": pd.Series(y).value_counts().to_dict(), "errors": errors, "elapsed": elapsed}

with st.sidebar:
    st.header("Platform settings")
    st.write("**Genres**")
    for g in GENRES: st.write(f"• {g.title()}")
    st.write("**Evaluation split:** 80% train / 20% test")
    st.write("**Random seed:** 42")
    if st.button("Clear training cache and retrain"):
        train_models.clear()
        extract_features_from_path.clear()
        st.rerun()

files = find_audio_files()
if not files:
    st.error("No audio files found yet.")
    st.markdown("""### Add your dataset
Copy your 500 GTZAN WAV files into this folder structure (relative to `app_platform2.py`):

```text
data/genres_original/
├── blues/       (100 audio files)
├── classical/   (100 audio files)
├── hiphop/      (100 audio files)
├── metal/       (100 audio files)
└── pop/         (100 audio files)
```

You can copy the existing `data/genres_original` folder from your current project into the `platform2` folder. Then restart the app.
""")
    st.stop()

signature = tuple((p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p, _ in files)
try:
    with st.spinner("Training Platform 2 models. The first run extracts features and may take a few minutes…"):
        bundle = train_models(signature)
except Exception as e:
    st.exception(e)
    st.stop()

results = bundle["results"]
metrics = st.columns(4)
metrics[0].metric("Readable tracks", bundle["total_tracks"])
metrics[1].metric("Genres found", len(bundle["counts"]))
metrics[2].metric("Feature dimensions", bundle["features"])
metrics[3].metric("Best test accuracy", f"{results['Test accuracy'].max():.1%}")

if bundle["errors"]:
    with st.expander(f"Audio files skipped due to read errors ({len(bundle['errors'])})"):
        st.write("\n".join(bundle["errors"][:100]))

st.subheader("1. Model performance")
st.dataframe(results.style.format({"Test accuracy": "{:.1%}", "Train time (s)": "{:.2f}", "Test prediction time (s)": "{:.4f}"}), use_container_width=True, hide_index=True)
fig = px.bar(results, x="Model", y="Test accuracy", text=results["Test accuracy"].map(lambda x: f"{x:.1%}"),
             title="Platform 2 — Test accuracy by model", range_y=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.subheader("2. Confusion matrix and detailed metrics")
selected = st.selectbox("Choose a model", list(bundle["models"].keys()))
model = bundle["models"][selected]
predictions = model.predict(bundle["X_test"])
labels = sorted(set(bundle["y_test"]) | set(predictions))
cm = confusion_matrix(bundle["y_test"], predictions, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", labels=dict(x="Predicted genre", y="Actual genre", color="Count"), title=f"{selected} — Confusion matrix")
st.plotly_chart(fig_cm, use_container_width=True)
report = classification_report(bundle["y_test"], predictions, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.subheader("3. Predict the genre of an audio file")
upload = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg"])
if upload:
    try:
        audio_bytes = upload.getvalue()
        y_audio, sr_audio = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=30)
        feature = extract_features_from_signal(y_audio, sr_audio).reshape(1, -1)
        model_name = st.selectbox("Prediction model", list(bundle["models"].keys()), key="prediction_model")
        chosen = bundle["models"][model_name]
        pred_genre = chosen.predict(feature)[0]
        st.success(f"Predicted genre: **{pred_genre.title()}** — using {model_name}")
        if hasattr(chosen, "predict_proba"):
            probs = chosen.predict_proba(feature)[0]
            class_names = chosen.classes_ if hasattr(chosen, "classes_") else chosen[-1].classes_
            # Pipeline exposes classes_ in recent sklearn versions; fallback to final estimator.
            if hasattr(chosen, "steps"):
                class_names = chosen.steps[-1][1].classes_
            p_df = pd.DataFrame({"Genre": class_names, "Probability": probs}).sort_values("Probability", ascending=False)
            st.plotly_chart(px.bar(p_df, x="Genre", y="Probability", range_y=[0, 1], title="Prediction confidence"), use_container_width=True)
        st.caption("Prediction is based on the features learned from your local dataset; confidence values are model scores, not guaranteed probabilities of correctness.")
    except Exception as e:
        st.error(f"Could not process this audio file: {e}")

st.subheader("4. Export results")
results_csv = results.to_csv(index=False).encode("utf-8")
st.download_button("Download accuracy results (CSV)", results_csv, "platform2_model_results.csv", "text/csv")
summary = {"platform": "Platform 2 — proposed models", "genres": GENRES, "split": "80/20 stratified", "random_seed": SEED,
           "track_count": bundle["total_tracks"], "genre_counts": bundle["counts"],
           "results": results.to_dict(orient="records"), "feature_version": FEATURE_VERSION}
st.download_button("Download experiment summary (JSON)", json.dumps(summary, indent=2), "platform2_experiment_summary.json", "application/json")
st.info("For your final report, use the same audio dataset and evaluation split for both platforms. Report the measured results from each run rather than assuming the new models will outperform the paper models.")
