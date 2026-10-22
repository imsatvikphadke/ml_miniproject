"""Step 2: the 1D Dilated CNN (paper Fig. IV.2) and its feature extractor."""
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def build_dcnn(input_shape, n_classes=len(C.GENRES), units=C.DEFAULT_UNITS,
               dropout=C.DROPOUT, norm_data=None) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="mfcc")
    norm = layers.Normalization(axis=-1, name="input_norm")   # per-MFCC-coefficient scaling
    if norm_data is not None:
        norm.adapt(norm_data)                                  # train data only
    x = norm(inp)
    for i, u in enumerate(units, 1):
        x = layers.Conv1D(u.filters, u.kernel, dilation_rate=u.dilation,
                          padding="same", activation="relu", name=f"dconv{i}")(x)
        x = layers.AveragePooling1D(u.pool, name=f"avgpool{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)       # unit output = "layer i" features
    x = layers.Flatten()(x)
    out = layers.Dense(n_classes, activation="softmax", name="softmax")(x)
    return keras.Model(inp, out, name="dilated_cnn")


def train_dcnn(X_train, y_train, X_test, y_test, epochs=C.EPOCHS, units=C.DEFAULT_UNITS,
               dropout=C.DROPOUT, save_to=C.CACHE_DIR / "dcnn.keras"):
    keras.utils.set_random_seed(C.SEED)
    model = build_dcnn(X_train.shape[1:], units=units, dropout=dropout, norm_data=X_train)
    model.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    to_cat = lambda y: keras.utils.to_categorical(y, len(C.GENRES))
    # The test set is only *monitored* (never used for early stopping / model selection).
    hist = model.fit(X_train, to_cat(y_train), validation_data=(X_test, to_cat(y_test)),
                     epochs=epochs, batch_size=C.BATCH_SIZE, verbose=2)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_to)
    with open(save_to.with_suffix(".history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f)
    return model


def load_dcnn(path=C.CACHE_DIR / "dcnn.keras") -> keras.Model:
    return keras.models.load_model(path)


def make_extractor(model: keras.Model) -> keras.Model:
    names = [l.name for l in model.layers]
    return keras.Model(model.input, [model.get_layer(n).output for n in ("bn1", "bn2") if n in names])


def extract_layer_features(model: keras.Model, X, batch_size=64) -> dict:
    """Flattened activations of each dilated-conv unit (Experiment B inputs)."""
    outs = make_extractor(model).predict(X, batch_size=batch_size, verbose=0)  # inference: no dropout
    return {f"dcnn_layer{i}": o.reshape(len(X), -1) for i, o in enumerate(outs, 1)}
