"""Pipeline CLI.

  python main.py extract --data-dir data/genres_original   # MFCC + 80/20 split
  python main.py train   [--epochs 100]                    # train the Dilated CNN
  python main.py evaluate                                  # Experiments A & B + baselines
  python main.py all
"""
import argparse
import numpy as np

from src import config as C
from src import features, dcnn, classical, evaluate


def cmd_extract(a):
    features.build_dataset(a.data_dir)


def cmd_train(a):
    Xtr, ytr, Xte, yte = features.load_dataset()
    dcnn.train_dcnn(Xtr, ytr, Xte, yte, epochs=a.epochs, dropout=a.dropout)


def cmd_evaluate(a):
    Xtr, ytr, Xte, yte = features.load_dataset()
    model = dcnn.load_dcnn()
    records = []

    # ---- Experiment A: the CNN's own genre predictions ----
    p_tr = model.predict(Xtr, verbose=0).argmax(1)
    p_te = model.predict(Xte, verbose=0).argmax(1)
    s = evaluate.score(yte, p_te)
    records.append(dict(experiment="A", features="mfcc_sequence", classifier="DCNN",
                        train_acc=float((p_tr == ytr).mean()), test_acc=s["accuracy"],
                        confusion=s["confusion"], report=s["report"]))

    # ---- Experiment B: classical classifiers on raw / PCA / CNN-layer features ----
    sets = classical.build_feature_sets(
        Xtr, Xte, dcnn.extract_layer_features(model, Xtr), dcnn.extract_layer_features(model, Xte))
    for fname, (ftr, fte) in sets.items():
        for cname, clf in classical.make_classifiers().items():
            print(f"fitting {cname} on {fname} {ftr.shape[1]} dims ...")
            clf.fit(ftr, ytr)
            s = evaluate.score(yte, clf.predict(fte))
            records.append(dict(experiment="B" if fname.startswith("dcnn") else "baseline",
                                features=fname, classifier=cname,
                                train_acc=float(clf.score(ftr, ytr)), test_acc=s["accuracy"],
                                confusion=s["confusion"], report=s["report"]))
    evaluate.save_results(records)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--data-dir", default=C.DATA_DIR); e.set_defaults(f=cmd_extract)
    t = sub.add_parser("train"); t.add_argument("--epochs", type=int, default=C.EPOCHS)
    t.add_argument("--dropout", type=float, default=C.DROPOUT); t.set_defaults(f=cmd_train)
    v = sub.add_parser("evaluate"); v.set_defaults(f=cmd_evaluate)
    al = sub.add_parser("all"); al.add_argument("--data-dir", default=C.DATA_DIR)
    al.add_argument("--epochs", type=int, default=C.EPOCHS)
    al.add_argument("--dropout", type=float, default=C.DROPOUT)
    al.set_defaults(f=lambda a: (cmd_extract(a), cmd_train(a), cmd_evaluate(a)))
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
