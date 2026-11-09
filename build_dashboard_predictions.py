"""Save the test-period predictions of the model the dashboard explains.

    python build_dashboard_predictions.py

Refits the XGBoost model of hull_aligned_design_eval.py (30 predictors plus centroid coordinates, seed 42) and
stops unless it reproduces the published hold-out R2, so the map layer cannot show a different model.
Writes outputs/dashboard_final/test_predictions_N4000.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import hull_aligned_design_eval as ev

DEST = ev.OUT / "dashboard_final" / "test_predictions_N4000.csv"
TOLERANCE = 1e-6


def main() -> None:
    panel = pd.read_csv(ev.PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = ev.choose_feature_players(panel)
    cols = players + ev.SPATIAL_COLUMNS
    train, val, test = ev.temporal_split(panel)
    model = ev.make_model("xgboost", ev.SEED, n_jobs=4)
    ev.fit_model("xgboost", model, panel.loc[train, cols], panel.loc[train, ev.Y_COL], panel.loc[val, cols], panel.loc[val, ev.Y_COL])
    predicted = np.asarray(model.predict(panel.loc[test, cols].astype(float)), float)
    observed = panel.loc[test, ev.Y_COL].to_numpy(float)
    r2 = float(r2_score(observed, predicted))
    published = json.loads((ev.OUT / "dashboard_final" / "feature_metrics_N4000.json").read_text(encoding="utf-8"))["predictive"]["r2"]
    if abs(r2 - published) > TOLERANCE:
        raise SystemExit(f"R2 {r2:.6f} does not reproduce the published {published:.6f}; nothing written")
    out = panel.loc[test, ["lsoa21cd", "lsoa21nm", "month"]].assign(observed=observed, predicted=predicted)
    out["month"] = out["month"].dt.strftime("%Y-%m")
    out.round(4).to_csv(DEST, index=False)
    print(f"R2 {r2:.6f} matches the published value; wrote {len(out)} rows to {DEST}")


if __name__ == "__main__":
    main()
