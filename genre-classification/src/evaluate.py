"""Step 4: metrics, confusion matrices, comparison tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config as C


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(C.GENRES))),
        "report": classification_report(y_true, y_pred, labels=range(len(C.GENRES)),
                                        target_names=C.GENRES, output_dict=True, zero_division=0),
    }


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(C.GENRES)), yticks=range(len(C.GENRES)),
           xticklabels=C.GENRES, yticklabels=C.GENRES,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_results(records: list, out_dir=C.RESULTS_DIR):
    """records: dicts with experiment, features, classifier, train_acc, test_acc, confusion, report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion").mkdir(exist_ok=True)
    rows = []
    for r in records:
        tag = f"{r['features']}__{r['classifier']}"
        plot_confusion(r["confusion"], f"{r['classifier']} on {r['features']}\n"
                       f"test acc {r['test_acc']:.2f}", out_dir / "confusion" / f"{tag}.png")
        rows.append({k: r[k] for k in ("experiment", "features", "classifier", "train_acc", "test_acc")})
        with open(out_dir / "confusion" / f"{tag}.report.json", "w") as f:
            json.dump(r["report"], f, indent=2)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_dir / "results.csv", index=False)

    # Paper-style Table V.3: rows = features x (train/test), columns = classifiers
    clf_df = df[df.experiment != "A"]
    table = clf_df.melt(id_vars=["features", "classifier"], value_vars=["train_acc", "test_acc"],
                        var_name="split", value_name="acc").pivot_table(
        index=["features", "split"], columns="classifier", values="acc")
    table.to_csv(out_dir / "table_v3_style.csv")
    print("\n=== Experiment A: DCNN's own predictions ===")
    print(df[df.experiment == "A"].to_string(index=False))
    print("\n=== Experiment B + baselines (train/test accuracy) ===")
    print(table.round(3).to_string())
    return df
