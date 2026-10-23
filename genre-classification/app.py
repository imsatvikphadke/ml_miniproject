"""Music Genre Classification dashboard.   Run:  streamlit run app.py"""
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from src import config as C
from src import inference as inf
from src.inference import FEATURE_NAMES, FEATURE_SETS, LABELS

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1250px}
div[data-testid="stMetric"]{background:rgba(120,120,160,.10);padding:14px 18px;border-radius:12px}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ load
if not ((C.CACHE_DIR / "dcnn.keras").exists() and (C.CACHE_DIR / "mfcc.npz").exists()):
    st.error("Trained artifacts not found. Put `dcnn.keras` and `mfcc.npz` in the `cache/` folder "
             "(or run `python main.py all`), then refresh.")
    st.stop()


@st.cache_resource(show_spinner="Loading the trained model and fitting classical classifiers (first run only)...")
def get_bundle():
    return inf.load_bundle()


b = get_bundle()
CLFS = ["LR", "GDA", "RF", "SVM"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🎵 About")
    st.write("Reproduction of **Combining CNN and Classical Algorithms for Music Genre "
             "Classification** (Stanford CS229, 2018).")
    st.markdown("""
**Pipeline**
1. GTZAN audio → MFCC (LibROSA)
2. Train a 1D **Dilated CNN**
3. *Exp. A:* CNN's own predictions
4. *Exp. B:* CNN layer activations → LR / GDA / RF / SVM
5. Compare accuracy & confusion matrices
""")
    st.caption(f"Genres: {', '.join(LABELS)}")
    st.caption(f"Train tracks: {len(b.ytr)} · Test tracks: {len(b.yte)}")
    st.caption(f"MFCC input: {C.N_MFCC} coeffs × {C.N_FRAMES} frames")
    st.markdown("[Original paper](https://cs229.stanford.edu/proj2018/report/19.pdf)")

st.title("Music Genre Classification")
st.caption("Dilated CNN as a feature extractor for classical classifiers")

tab_demo, tab_res, tab_train, tab_space = st.tabs(
    ["🎧 Live demo", "📊 Results", "🧠 Model & training", "🌌 Feature space"])

# ================================================================== LIVE DEMO
with tab_demo:
    source = st.radio("Audio source", ["Held-out test track", "Upload your own audio"], horizontal=True)
    X, true_label, audio_bytes = None, None, None

    if source == "Held-out test track":
        names = [str(f) for f in b.files_test]
        if "pick" not in st.session_state:
            st.session_state["pick"] = names[0]
        c1, c2 = st.columns([4, 1])
        c2.write("")
        c2.button("🎲 Random track", width="stretch",
                  on_click=lambda: st.session_state.update(pick=random.choice(names)))
        c1.selectbox("Test track (never seen during training)", names, key="pick")
        i = names.index(st.session_state["pick"])
        X, true_label = b.Xte[i:i + 1], int(b.yte[i])
    else:
        up = st.file_uploader("Upload a music clip (wav / mp3 / ogg / flac)", type=["wav", "mp3", "ogg", "flac"])
        if up is not None:
            try:
                y = inf.load_audio(up.getvalue())
                dur = len(y) / C.SR
                st.audio(up.getvalue())
                start = 0.0
                if dur > C.DURATION + 1:
                    start = st.slider("Start of the 30-second analysis window (s)", 0.0,
                                      float(dur - C.DURATION), 0.0, 1.0)
                elif dur < C.DURATION:
                    st.info(f"Clip is {dur:.1f}s; it will be zero-padded to 30s (accuracy may drop).")
                X = inf.mfcc_window(y, start)[None]
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that audio file: {e}")
        else:
            st.info("Upload a clip to classify it. The model was trained on 30 s GTZAN clips of "
                    "classical, hip-hop, metal, pop and blues.")

    if X is not None:
        res = inf.predict(b, X)
        probs = res["cnn_probs"][0]
        top = int(probs.argmax())
        st.divider()

        votes = [res["preds"][(f, c)][0] for f in ("dcnn_layer1", "dcnn_layer2") for c in CLFS]
        vote, n_votes = Counter(votes).most_common(1)[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("CNN prediction (Exp. A)", LABELS[top], f"{probs[top]:.0%} confidence", delta_color="off")
        m2.metric("Classical consensus (Exp. B)", LABELS[vote], f"{n_votes}/{len(votes)} models agree",
                  delta_color="off")
        if true_label is not None:
            ok = top == true_label
            m3.metric("True genre", LABELS[true_label], "CNN correct ✓" if ok else "CNN wrong ✗",
                      delta_color="normal" if ok else "inverse")
        else:
            m3.metric("True genre", "unknown", "uploaded audio", delta_color="off")

        left, right = st.columns([2, 3])
        with left:
            st.subheader("CNN probabilities")
            fig = go.Figure(go.Bar(x=probs, y=LABELS, orientation="h",
                                   marker_color=["#4f8cff" if i == top else "#9aa5b8" for i in range(5)],
                                   text=[f"{p:.1%}" for p in probs], textposition="outside"))
            fig.update_layout(height=300, margin=dict(l=0, r=30, t=10, b=0),
                              xaxis=dict(range=[0, 1.1], title="probability"), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        with right:
            st.subheader("MFCC the model sees")
            fig = px.imshow(X[0].T, aspect="auto", origin="lower", color_continuous_scale="Viridis",
                            labels=dict(x="Frame (≈23 ms each)", y="MFCC coefficient", color="value"))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Every classical classifier, on every feature set")
        tbl = pd.DataFrame({c: [LABELS[res["preds"][(f, c)][0]] for f in FEATURE_SETS] for c in CLFS},
                           index=[FEATURE_NAMES[f] for f in FEATURE_SETS])
        if true_label is not None:
            fn = lambda v: ("background-color:rgba(46,160,67,.35)" if v == LABELS[true_label]
                            else "background-color:rgba(220,60,60,.30)")
            tbl = tbl.style.map(fn) if hasattr(tbl.style, "map") else tbl.style.applymap(fn)
            st.caption("Green = correct, red = wrong.")
        st.dataframe(tbl, width="stretch")

# ================================================================== RESULTS
with tab_res:
    cnn_acc = b.test_acc[("mfcc_sequence", "DCNN")]
    best = max(b.clfs, key=lambda k: b.test_acc[k])
    baseline_best = max((k for k in b.clfs if not k[0].startswith("dcnn")), key=lambda k: b.test_acc[k])
    cnn_feat_best = max((k for k in b.clfs if k[0].startswith("dcnn")), key=lambda k: b.test_acc[k])
    a, bb, c, d = st.columns(4)
    a.metric("DCNN test accuracy", f"{cnn_acc:.2f}")
    bb.metric("Best on raw / PCA features", f"{b.test_acc[baseline_best]:.2f}",
              f"{baseline_best[1]} · {FEATURE_NAMES[baseline_best[0]]}", delta_color="off")
    c.metric("Best on CNN features", f"{b.test_acc[cnn_feat_best]:.2f}",
             f"{cnn_feat_best[1]} · {FEATURE_NAMES[cnn_feat_best[0]]}", delta_color="off")
    d.metric("Test set size", f"{len(b.yte)} tracks")
    st.caption("With only 100 test tracks, one standard error is ≈3–4 points; "
               "small differences are within noise.")

    acc = pd.DataFrame({c: [b.test_acc[(f, c)] for f in FEATURE_SETS] for c in CLFS},
                       index=[FEATURE_NAMES[f] for f in FEATURE_SETS])
    gap = pd.DataFrame({c: [b.train_acc[(f, c)] - b.test_acc[(f, c)] for f in FEATURE_SETS] for c in CLFS},
                       index=acc.index)
    h1, h2 = st.columns(2)
    with h1:
        st.subheader("Test accuracy")
        fig = px.imshow(acc, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with h2:
        st.subheader("Overfitting gap (train − test)")
        fig = px.imshow(gap, text_auto=".2f", color_continuous_scale="Reds", zmin=0, zmax=0.5, aspect="auto")
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Experiment A vs B")
    long = acc.reset_index(names="features").melt("features", var_name="classifier", value_name="test accuracy")
    fig = px.bar(long, x="features", y="test accuracy", color="classifier", barmode="group")
    fig.add_hline(y=cnn_acc, line_dash="dash", annotation_text=f"DCNN alone = {cnn_acc:.2f}")
    fig.update_layout(height=380, yaxis=dict(range=[0, 1]), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Confusion matrix & per-class metrics")
    options = [("mfcc_sequence", "DCNN")] + [(f, c) for f in FEATURE_SETS for c in CLFS]
    sel = st.selectbox("Model", options, index=0,
                       format_func=lambda k: "DCNN (Experiment A)" if k[1] == "DCNN"
                       else f"{k[1]} on {FEATURE_NAMES[k[0]]}")
    pred = b.test_pred[sel]
    cm = confusion_matrix(b.yte, pred, labels=range(5))
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        fig = px.imshow(cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with cc2:
        rep = pd.DataFrame(classification_report(b.yte, pred, labels=range(5), target_names=LABELS,
                                                 output_dict=True, zero_division=0)).T
        st.dataframe(rep.round(2), width="stretch")
        st.caption(f"Test accuracy: {b.test_acc[sel]:.2f} · train accuracy: {b.train_acc[sel]:.2f}")

# ================================================================== MODEL & TRAINING
with tab_train:
    hist_path = C.CACHE_DIR / "dcnn.history.json"
    if hist_path.exists():
        h = json.load(open(hist_path))
        ep = list(range(1, len(h["loss"]) + 1))
        t1, t2 = st.columns(2)
        for col, key, title in ((t1, "accuracy", "Accuracy"), (t2, "loss", "Loss")):
            fig = go.Figure()
            fig.add_scatter(x=ep, y=h[key], name="train")
            fig.add_scatter(x=ep, y=h[f"val_{key}"], name="test (monitor only)")
            fig.update_layout(title=title, height=330, xaxis_title="epoch", margin=dict(l=0, r=0, t=40, b=0))
            col.plotly_chart(fig, width="stretch")
        st.caption("The test split was only monitored while training; it was not used for early stopping "
                   "or model selection.")
    else:
        st.info("Training history not found (`cache/dcnn.history.json`).")

    st.subheader("Architecture")
    rows = []
    for l in b.model.layers:
        try:
            shape = str(tuple(l.output.shape[1:]))
        except Exception:  # noqa: BLE001
            shape = "-"
        rows.append({"layer": l.name, "type": type(l).__name__, "output shape": shape, "params": l.count_params()})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Total parameters: {b.model.count_params():,}")

    st.subheader("Hyperparameters")
    u = C.DEFAULT_UNITS
    st.table(pd.DataFrame({
        "setting": ["Unit 1 (filters, kernel, dilation, pool)", "Unit 2 (filters, kernel, dilation, pool)",
                    "Optimizer", "Loss", "Dropout", "Batch size", "Epochs", "PCA components", "RF max depth"],
        "value": [f"{u[0].filters}, {u[0].kernel}, {u[0].dilation}, {u[0].pool}",
                  f"{u[1].filters}, {u[1].kernel}, {u[1].dilation}, {u[1].pool}",
                  f"Adam (lr={C.LEARNING_RATE})", "Categorical cross-entropy", C.DROPOUT,
                  C.BATCH_SIZE, C.EPOCHS, C.PCA_COMPONENTS, C.RF_MAX_DEPTH]}).astype(str))

# ================================================================== FEATURE SPACE
with tab_space:
    st.write("Each dot is a track, projected to 3D with PCA. Compare how well genres separate in "
             "raw MFCC space vs the CNN's learned layers (paper, Fig. IV.3).")
    fsel = st.selectbox("Feature set", FEATURE_SETS[0:1] + FEATURE_SETS[2:], format_func=FEATURE_NAMES.get)
    show = st.radio("Points", ["Test set", "Train + test"], horizontal=True)
    Ftr, Fte = b.feats_tr[fsel], b.feats_te[fsel]
    pca3 = PCA(n_components=3, random_state=C.SEED).fit(Ftr)
    if show == "Test set":
        Z, y, split = pca3.transform(Fte), b.yte, ["test"] * len(b.yte)
    else:
        Z = np.vstack([pca3.transform(Ftr), pca3.transform(Fte)])
        y = np.concatenate([b.ytr, b.yte]); split = ["train"] * len(b.ytr) + ["test"] * len(b.yte)
    df = pd.DataFrame(Z, columns=["PC1", "PC2", "PC3"])
    df["genre"] = [LABELS[i] for i in y]; df["split"] = split
    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color="genre", symbol="split",
                        category_orders={"genre": LABELS}, opacity=0.85)
    fig.update_traces(marker_size=4)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Variance explained by the 3 components: {pca3.explained_variance_ratio_.sum():.1%}")
