"""Every model_factory candidate must satisfy GeoSMILE's fit/predict contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_factory import MODEL_NAMES, fit_model, make_model

SEED = 0


def _synthetic_split(n_train=60, n_val=20, n_test=20, n_features=4):
    rng = np.random.default_rng(SEED)

    def make(n):
        X = pd.DataFrame(
            rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)]
        )
        y = rng.poisson(lam=3.0, size=n).astype(float)
        return X, y

    return make(n_train), make(n_val), make(n_test)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fit_predict_contract(model_name):
    (X_train, y_train), (X_val, y_val), (X_test, _) = _synthetic_split()
    model = make_model(model_name, SEED)
    fit_model(model_name, model, X_train, y_train, X_val, y_val)
    pred = np.asarray(model.predict(X_test), dtype=float)
    assert pred.shape == (len(X_test),)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= -1e-6)
