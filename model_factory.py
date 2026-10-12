"""Model construction/fitting for GeoSMILE's black-box candidates.

GeoSMILE's SMILE attribution machinery (see `geosmile_hull_twobranch.py`) only ever
calls `.predict()` on the underlying model, so any regressor exposing the standard
scikit-learn `fit(X, y)` / `predict(X)` interface can be explained by it. This module
centralizes construction of the published model (XGBoost) alongside six alternative
"not self-explainable" candidates, so scripts only need a model name string.

None of the alternatives are hyperparameter-tuned: they are sized to be reasonable,
comparable-capacity baselines for screening against XGBoost, not optimized for best
possible accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import geosmile_hull_twobranch as geosmile_hull

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc


MODEL_NAMES = ("xgboost", "lightgbm", "catboost", "mlp", "svr", "gp", "dl", "ann", "cnn", "transformer")
BOOSTED_MODEL_NAMES = ("xgboost", "lightgbm", "catboost")


class _BaseKerasRegressor(BaseEstimator, RegressorMixin):
    """Shared fit(X, y)/predict(X) scaffold for every Keras candidate in this
    module -- scaling/target-transform live outside (see
    _wrap_scaled_count_regressor), so this handles only: seeding, building
    the architecture (via _build_model, overridden per subclass), compiling,
    early-stopping-on-validation-split, and prediction. Subclasses
    BaseEstimator/RegressorMixin (get_params/set_params from constructor args
    named identically to attributes) purely so TransformedTargetRegressor can
    clone() it -- no sklearn training logic is inherited or used.
    """

    def __init__(self, seed: int, epochs: int = 200, batch_size: int = 32):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None

    def _build_model(self, n_features: int):
        raise NotImplementedError

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(self.seed)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        model = self._build_model(X.shape[1])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse")
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        model.fit(
            X,
            y,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0,
        )
        self._model = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        pred = self._model.predict(X, verbose=0)
        return np.asarray(pred, dtype=float).reshape(-1)


class KerasCountRegressor(_BaseKerasRegressor):
    """Small feedforward Keras network -- the original "deep learning"
    candidate, distinct from the sklearn MLPRegressor ("mlp"). Both are
    shallow-ish feedforward nets in spirit, but this one is deeper (3 hidden
    layers vs. 2), uses dropout for regularization, and trains with Keras'
    own early-stopping-on-validation-split rather than sklearn's.
    """

    def _build_model(self, n_features: int):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )


class KerasANNRegressor(_BaseKerasRegressor):
    """A second, deliberately distinct plain feedforward ANN: wider (256/128
    vs. 128/64) and adds LayerNormalization, so it is not just a relabelling
    of "dl" -- a genuinely different capacity/regularization point in the
    same dense-network family. Uses LayerNormalization rather than
    BatchNormalization deliberately: BatchNorm's train/inference statistics
    mismatch caused catastrophic divergence at predict time on this dataset
    (held-out R2 in the hundreds-negative) even though training looked fine;
    LayerNormalization has no such running-statistics state.
    """

    def _build_model(self, n_features: int):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )


class KerasCNNRegressor(_BaseKerasRegressor):
    """1D-convolutional adaptation for flat tabular input: reshapes the
    feature vector to a length-n_features, single-channel "sequence" and
    applies Conv1D over it. There is no genuine spatial/sequential order
    across these 30 features -- this is the standard (if debated)
    tabular-CNN heuristic, included as a plausible baseline a reviewer might
    ask about, not because a spatial CNN is a principled fit for this branch
    (a rasterized-LSOA-grid CNN on the Geo branch would be the principled
    version, and is a bigger architectural change than "swap the model",
    not attempted here).
    """

    def _build_model(self, n_features: int):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Reshape((n_features, 1)),
                tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1),
            ]
        )


class _PositionalEmbedding(tf.keras.layers.Layer):
    """Trainable per-column position embedding, added to the per-feature
    token embedding below -- multi-head attention is permutation-invariant,
    so without this the transformer could not tell which original column
    each token came from.
    """

    def build(self, input_shape):
        n_positions, embed_dim = input_shape[-2], input_shape[-1]
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(n_positions, embed_dim), initializer="random_normal", trainable=True
        )

    def call(self, x):
        return x + self.pos_embed


class KerasTransformerRegressor(_BaseKerasRegressor):
    """Lightweight self-attention adaptation for tabular input: each of the
    n_features columns is treated as a "token", linearly projected to a
    small embedding with a learned positional embedding added, then passed
    through one transformer-encoder block (multi-head self-attention +
    residual + layer norm + a small feedforward block) before pooling to a
    scalar prediction. Deliberately small (embed_dim=16, 2 heads, 1 block)
    for a ~3,000-row training set -- not an NLP-scale transformer.
    """

    EMBED_DIM = 16
    NUM_HEADS = 2
    FF_DIM = 32

    def _build_model(self, n_features: int):
        inputs = tf.keras.layers.Input(shape=(n_features,))
        x = tf.keras.layers.Reshape((n_features, 1))(inputs)
        x = tf.keras.layers.Dense(self.EMBED_DIM)(x)
        x = _PositionalEmbedding()(x)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=self.NUM_HEADS, key_dim=self.EMBED_DIM
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_out)
        ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.FF_DIM, activation="relu"),
                tf.keras.layers.Dense(self.EMBED_DIM),
            ]
        )
        x = tf.keras.layers.LayerNormalization()(x + ff(x))
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


def _floor_expm1(z: np.ndarray) -> np.ndarray:
    """Invert log1p and floor at 0 -- crime counts cannot be negative."""
    return np.clip(np.expm1(z), 0, None)


def _wrap_scaled_count_regressor(estimator) -> Pipeline:
    target = TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=_floor_expm1
    )
    return Pipeline([("scale", StandardScaler()), ("est", target)])


def make_model(model_name: str, seed: int, n_jobs: int = -1):
    if model_name == "xgboost":
        model = geosmile_hull.make_model(seed)
        model.set_params(n_jobs=n_jobs)
        return model
    if model_name == "lightgbm":
        return LGBMRegressor(
            objective="poisson",
            n_estimators=500,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if model_name == "catboost":
        return CatBoostRegressor(
            loss_function="Poisson",
            iterations=500,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=2.0,
            subsample=0.85,
            bootstrap_type="Bernoulli",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "mlp":
        return _wrap_scaled_count_regressor(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                early_stopping=True,
                random_state=seed,
            )
        )
    if model_name == "svr":
        return _wrap_scaled_count_regressor(SVR(kernel="rbf", C=10.0, epsilon=0.5))
    if model_name == "gp":
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
        return _wrap_scaled_count_regressor(
            GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        )
    if model_name == "dl":
        return _wrap_scaled_count_regressor(KerasCountRegressor(seed))
    if model_name == "ann":
        return _wrap_scaled_count_regressor(KerasANNRegressor(seed))
    if model_name == "cnn":
        return _wrap_scaled_count_regressor(KerasCNNRegressor(seed))
    if model_name == "transformer":
        return _wrap_scaled_count_regressor(KerasTransformerRegressor(seed))
    raise ValueError(f"Unknown model_name {model_name!r}; expected one of {MODEL_NAMES}")


def fit_model(model_name: str, model, X_train, y_train, X_val, y_val):
    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "lightgbm":
        # LightGBM's sklearn wrapper controls verbosity via the constructor's
        # verbose=-1 (set in make_model), not a fit() kwarg.
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    elif model_name == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train)
    return model
