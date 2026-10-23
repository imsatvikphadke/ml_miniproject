# Music Genre Classification: Dilated CNN + Classical Algorithms

Reproduction of *"Combining CNN and Classical Algorithms for Music Genre Classification"*
(Li, Xue, Zhang, Stanford CS229, 2018): https://cs229.stanford.edu/proj2018/report/19.pdf

## Pipeline
| Step | Module | What it does |
|---|---|---|
| 1 | `src/features.py` | LibROSA: 22.05 kHz, 30 s, 20 MFCC x 1290 frames; stratified 400/100 split (seed 42) |
| 2 | `src/dcnn.py` | 1D Dilated CNN: [Conv1D(dilated) -> AvgPool -> Dropout -> BatchNorm] x 2 -> Softmax. Adam 1e-3, categorical cross-entropy |
| 3 | `src/classical.py` | Softmax LR, GDA (LDA), Random Forest (depth 7), SVM (RBF); PCA to 50 comps |
| 4 | `src/evaluate.py` | Train/test accuracy, confusion matrices, per-class metrics, Table V.3-style summary |

**Experiment A:** the CNN's own predictions.
**Experiment B:** classical classifiers on (1) raw flattened MFCC (25,800), (2) PCA-50, (3) DCNN layer-1 activations, (4) DCNN layer-2 activations. (1) and (2) are the baselines.

## Run
```bash
pip install -r requirements.txt
# Download GTZAN (e.g. Kaggle "andradaolteanu/gtzan-dataset-music-genre-classification")
# so that files look like data/genres_original/blues/blues.00000.wav
python main.py all                # or: extract -> train -> evaluate
```
Outputs land in `results/` (`results.csv`, `table_v3_style.csv`, `confusion/*.png`).

## Notes / known deviations from the paper
- The paper doesn't specify its framework; this uses TensorFlow/Keras.
- CNN config = the "simplest" row of Table V.2 (8,8,4,32 / 16,4,2,8). Layer-1 features have 320 dims (matches paper); layer-2 has 80 dims here vs 48 in the paper, whose exact layer-2 config isn't fully recoverable. Edit `DEFAULT_UNITS` in `src/config.py` to experiment.
- Paper-unspecified choices: epochs (100), batch size (32), RF trees (100), input normalisation for the CNN (per-coefficient, fitted on train). Baselines use unscaled MFCC, as in the paper.
- The test set is only monitored during training, never used for early stopping or selection. With 100 test tracks, expect a few points of run-to-run noise.

## Dashboard (demo)
```bash
streamlit run app.py
```
Needs `cache/dcnn.keras`, `cache/mfcc.npz` (and optionally `cache/dcnn.history.json`) from training.
Tabs: **Live demo** (pick a held-out test track or upload audio -> CNN probabilities, MFCC heatmap,
all 16 classical predictions), **Results** (accuracy heatmaps, overfitting gap, confusion matrices),
**Model & training** (curves, architecture), **Feature space** (3D PCA of raw vs CNN features).
The first launch fits the classical classifiers (~1 min) and saves `cache/classical.joblib`;
delete that file to refit.
