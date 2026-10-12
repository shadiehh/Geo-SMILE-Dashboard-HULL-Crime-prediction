"""Predictive-performance-only check (no GeoSMILE fitting) for one or more
candidate models -- the cheap first step of the R2>0.7 gate used by every
GeoSMILE-fitting script in this project (surrogate_comparison.py onward).
Unlike eval_by_model.py (which runs the full, expensive GeoSMILE pipeline
unconditionally), this only ever calls .fit()/.predict() once per branch --
safe to run for a new candidate before deciding whether its predictive fit
justifies the expensive attribution step at all.

Writes into (merges with) outputs/eval_by_model/predictive_performance.csv,
the same file every later gated script reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_by_model import predictive_performance
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, choose_feature_players, temporal_split
from model_factory import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "eval_by_model"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)

    rows = []
    for model_name in args.models:
        _log(f"[{model_name}] predictive performance (Feature + Geo)...")
        model_rows = predictive_performance(panel, players, train, validation, test, model_name)
        for r in model_rows:
            _log(f"[{model_name}] {r['branch']}: R2={r['r2']:.4f} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}")
        rows.extend(model_rows)

    new_df = pd.DataFrame(rows)
    out_path = DEST / "predictive_performance.csv"
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        df = pd.concat([old_df[~old_df["model"].isin(args.models)], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {out_path}")


if __name__ == "__main__":
    main()
