"""Step 3: classical classifiers + feature-set construction."""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from . import config as C


def make_classifiers():
    return {
        "LR":  LogisticRegression(max_iter=2000),   # multinomial softmax (lbfgs)
        "GDA": LinearDiscriminantAnalysis(),        # Gaussians, shared covariance = GDA
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=C.RF_MAX_DEPTH,
                                      criterion="gini", random_state=C.SEED, n_jobs=-1),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=C.SEED),  # one-vs-one, RBF
    }


def build_feature_sets(X_train_raw, X_test_raw, cnn_train: dict, cnn_test: dict) -> dict:
    """Returns {name: (train_features, test_features)} for the four paper representations."""
    flat_tr = X_train_raw.reshape(len(X_train_raw), -1)   # 20 x 1290 = 25,800
    flat_te = X_test_raw.reshape(len(X_test_raw), -1)
    pca = PCA(n_components=min(C.PCA_COMPONENTS, len(flat_tr)), random_state=C.SEED).fit(flat_tr)  # fit on train only
    sets = {
        "raw_mfcc": (flat_tr, flat_te),
        "pca_mfcc": (pca.transform(flat_tr), pca.transform(flat_te)),
    }
    for name in cnn_train:
        sets[name] = (cnn_train[name], cnn_test[name])
    return sets
