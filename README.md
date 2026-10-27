# Music Genre Classifier — Platform 2

A separate Streamlit platform for the proposed model family. It trains and compares three models that are different from the four classical classifiers listed in the reference paper:

- K-Nearest Neighbors (KNN)
- Extra Trees Classifier
- Histogram Gradient Boosting Classifier

The reference paper uses a Dilated CNN and Logistic Regression, GDA, Random Forest, and SVM. Platform 2 uses summary audio features (MFCC, chroma, spectral centroid/bandwidth/rolloff, zero-crossing rate and RMS) and the same five genre labels.

## 1. Dataset folder

Place the 500 GTZAN audio files under this folder, with 100 files per genre:

```text
platform2/
├── app_platform2.py
├── requirements.txt
├── README.md
└── data/
    └── genres_original/
        ├── blues/
        ├── classical/
        ├── hiphop/
        ├── metal/
        └── pop/
```

You can copy `data/genres_original` from the existing paper-replication project into this folder. WAV is recommended; MP3, FLAC and OGG are also supported.

## 2. Windows setup

Open a terminal in this `platform2` folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app_platform2.py
```

If you already use a working virtual environment for the main project, you can activate that environment instead and install `requirements.txt` there.

Streamlit prints a local address, usually `http://localhost:8501`. Open it in your browser. If your original platform is already using port 8501, run this platform on another port:

```powershell
streamlit run app_platform2.py --server.port 8502
```

Then open `http://localhost:8502`.

## 3. What the platform includes

- First-run feature extraction and model training, cached for subsequent visits
- Stratified 80/20 train/test split with fixed random seed 42
- Test accuracy and training/prediction timings
- Confusion matrix and classification report (precision, recall, F1-score)
- Upload-audio genre prediction
- CSV and JSON downloads of measured experiment results

## Important experimental note

The models are trained locally on your supplied dataset; this ZIP intentionally does not include the large audio dataset. Results will only appear after the dataset is copied into the expected folder. Do not claim the new models are better until you compare measured results. For a fair comparison, use the same audio files and preferably the same fixed train/test filenames as the replication platform. The summary feature representation differs from the paper's flattened MFCC and DCNN activations, so document this as your proposed approach.
