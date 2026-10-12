"""geosmile.py -- a clean, dataset-agnostic implementation of GeoSMILE.

GeoSMILE extends the SMILE explainability framework (Aslansefat et al.) with
an explicit spatial branch and distributional (Wasserstein-1) perturbation
responses. This module implements the design validated on the Hull
crime-prediction case study, generalised to any tabular dataset:

  * ONE unified predictive model is trained on all predictors together --
    spatial (coordinate) columns and non-spatial columns alike. GeoSMILE
    explains that single fitted model; it does not fit a separate model per
    branch.
  * The explanation is split into two independent perturbation branches:
      - Feature branch: players are the non-spatial predictors. A
        perturbation masks a random subset of them to a baseline value,
        holding the spatial columns at each row's true value.
      - Geo branch: players are spatial units. If rows are grouped into
        named spatial units (e.g. a zone/region id), each unique unit is one
        player and a perturbation masks a random subset of units' spatial
        columns to a baseline value. Otherwise each row is treated as its
        own spatial player.
  * The surrogate is a kernel-weighted Ridge (alpha=1) fit on
    perturbation-mask -> Wasserstein-1-response pairs, with the kernel
    (rbf / laplace / inverse) chosen by validation-split R^2. The default
    perturbation budget is N=4,000 for both branches.
  * Evaluation covers the five ATT properties: Accuracy (vs. an external
    reference, optional), Fidelity, Faithfulness, Stability (vs. a
    user-supplied modified dataset), and Consistency (across independently
    seeded reruns) -- see the `evaluation` section below.

Minimal usage
-------------
    from geosmile import GeoSMILE

    explainer = GeoSMILE(
        estimator=SomeSklearnRegressor(...),   # unfitted; fit internally on
        X=X, y=y,                              # ALL columns together
        feature_columns=non_spatial_cols,
        spatial_columns=["easting", "northing"],
        spatial_unit_id="zone_id",             # optional grouping column
        seed=42,
    )
    feature_result = explainer.explain_feature_branch(n_perturb=4000)
    geo_result = explainer.explain_geo_branch(n_perturb=4000)

    print(feature_result.importance.head())     # signed + abs coefficients
    print(feature_result.fidelity)               # R2, weighted R2, WMAE, ...

    faith = explainer.faithfulness(feature_result, k=10)
    cons = explainer.consistency("feature", seeds=[42, 52, 62, 72, 82])

See the `__main__` block at the bottom for a runnable synthetic-data demo.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

KernelName = Literal["rbf", "laplace", "inverse"]
KERNELS: tuple[KernelName, ...] = ("rbf", "laplace", "inverse")

RANDOM_SEED_DEFAULT = 42
N_PERTURB_DEFAULT = 4000
RIDGE_ALPHA = 1.0
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2  # remainder (0.2) is the held-out test split


# --------------------------------------------------------------------------
# Core perturbation / surrogate machinery
# --------------------------------------------------------------------------

def mask_pool(n_masks: int, n_players: int, rng: np.random.Generator) -> np.ndarray:
    """n_masks x n_players binary matrix (1 = player kept, 0 = masked).

    Row 0 is all-ones (nothing masked) and row 1 is all-zeros (everything
    masked) so both extremes of the perturbation space are always sampled.
    Duplicate rows are removed, so the returned pool may be slightly smaller
    than n_masks when n_players is tiny.
    """
    masks = rng.integers(0, 2, size=(n_masks, n_players), dtype=np.int8)
    masks[0] = 1
    masks[1] = 0
    return np.unique(masks, axis=0)


def apply_kernel(distance: np.ndarray, kind: KernelName, sigma: float) -> np.ndarray:
    d = np.asarray(distance, float)
    sigma = max(float(sigma), 1e-12)
    if kind == "rbf":
        return np.exp(-((d / sigma) ** 2))
    if kind == "laplace":
        return np.exp(-(d / sigma))
    if kind == "inverse":
        return 1.0 / (1.0 + d / sigma)
    raise ValueError(f"Unknown kernel {kind!r}")


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    a, b = int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)
    return order[:a], order[a:b], order[b:]


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_true, y_pred, weights = map(lambda a: np.asarray(a, float), (y_true, y_pred, weights))
    if weights.sum() <= 0:
        return float("nan")
    mean_w = np.average(y_true, weights=weights)
    ss_res = float(np.sum(weights * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y_true - mean_w) ** 2))
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def fidelity_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """R2 / weighted R2 / MAE / WMAE / RMSE / WRMSE, per the Fidelity appendix."""
    y_true, y_pred, weights = map(lambda a: np.asarray(a, float), (y_true, y_pred, weights))
    resid = y_true - y_pred
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "Weighted_R2": weighted_r2_score(y_true, y_pred, weights),
        "MAE": float(np.mean(np.abs(resid))),
        "WMAE": float(np.average(np.abs(resid), weights=weights)),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "WRMSE": float(np.sqrt(np.average(resid ** 2, weights=weights))),
    }


@dataclass
class SurrogateFit:
    """A fitted kernel-weighted Ridge surrogate plus its evaluation pool."""

    coefficients: np.ndarray
    player_names: list[str]
    best_kernel: KernelName
    sigma: float
    validation_r2_by_kernel: dict[str, float]
    fidelity: dict[str, float]
    masks: np.ndarray
    responses: np.ndarray
    distances: np.ndarray

    @property
    def importance(self) -> pd.DataFrame:
        coef = self.coefficients
        return (
            pd.DataFrame({"player": self.player_names, "coefficient": coef, "abs_coefficient": np.abs(coef)})
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )


def fit_weighted_ridge_surrogate(
    masks: np.ndarray,
    responses: np.ndarray,
    player_names: list[str],
    *,
    seed: int,
    kernels: Sequence[KernelName] = KERNELS,
) -> SurrogateFit:
    """Kernel-weighted Ridge, matching the validated Hull design exactly:
    distance = fraction of players masked out; sigma = max(median train
    distance, 1/n_players); kernel chosen by validation R^2; final surrogate
    refit on train+validation; fidelity reported on the untouched test split.
    """
    n = len(masks)
    distances = 1.0 - masks.mean(axis=1)
    tr, va, te = _split_indices(n, seed)
    sigma = max(float(np.median(distances[tr])), 1.0 / masks.shape[1])

    scores: dict[str, float] = {}
    for kernel in kernels:
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(masks[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], ridge.predict(masks[va])) if len(va) > 1 else float("nan")
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best_kernel: KernelName = max(finite, key=finite.get) if finite else kernels[0]  # type: ignore[arg-type]

    trva = np.concatenate([tr, va])
    final = Ridge(alpha=RIDGE_ALPHA)
    final.fit(masks[trva], responses[trva], sample_weight=apply_kernel(distances[trva], best_kernel, sigma))

    w_te = np.clip(apply_kernel(distances[te], best_kernel, sigma), 1e-8, None)
    pred_te = final.predict(masks[te])
    fid = fidelity_metrics(responses[te], pred_te, w_te) if len(te) > 1 else {}

    return SurrogateFit(
        coefficients=final.coef_,
        player_names=player_names,
        best_kernel=best_kernel,
        sigma=sigma,
        validation_r2_by_kernel=scores,
        fidelity=fid,
        masks=masks,
        responses=responses,
        distances=distances,
    )


# --------------------------------------------------------------------------
# GeoSMILE explainer
# --------------------------------------------------------------------------

@dataclass
class BranchResult:
    branch: Literal["feature", "geo"]
    surrogate: SurrogateFit
    n_perturb_requested: int
    seed: int

    @property
    def importance(self) -> pd.DataFrame:
        return self.surrogate.importance

    @property
    def fidelity(self) -> dict[str, float]:
        return self.surrogate.fidelity


class GeoSMILE:
    """Explains ONE fitted predictive model along a Feature branch and a Geo
    branch. The model is trained on `feature_columns + spatial_columns`
    together (the "unified model" design), never on one branch alone.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature_columns: Sequence[str],
        spatial_columns: Sequence[str],
        *,
        model=None,
        estimator=None,
        spatial_unit_id: str | None = None,
        baseline: pd.Series | None = None,
        seed: int = RANDOM_SEED_DEFAULT,
    ) -> None:
        if model is None and estimator is None:
            raise ValueError("Pass either a fitted `model` or an unfitted `estimator` to fit internally.")
        self.X = X.reset_index(drop=True)
        self.y = np.asarray(y).ravel()
        self.feature_columns = list(feature_columns)
        self.spatial_columns = list(spatial_columns)
        self.all_columns = self.feature_columns + self.spatial_columns
        self.spatial_unit_id = spatial_unit_id
        self.seed = seed

        if model is not None:
            self.model = model
        else:
            self.model = estimator
            self.model.fit(self.X[self.all_columns], self.y)

        self.baseline = baseline if baseline is not None else self.X[self.all_columns].median()
        self.predictions = np.asarray(self.model.predict(self.X[self.all_columns]), float)

    # ---- Feature branch -------------------------------------------------

    def _perturb_feature(self, mask_row: np.ndarray) -> np.ndarray:
        perturbed = self.X[self.all_columns].copy()
        for j in np.flatnonzero(mask_row == 0):
            col = self.feature_columns[j]
            perturbed[col] = self.baseline[col]
        return np.asarray(self.model.predict(perturbed), float)

    def explain_feature_branch(
        self, n_perturb: int = N_PERTURB_DEFAULT, seed: int | None = None
    ) -> BranchResult:
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        masks = mask_pool(n_perturb, len(self.feature_columns), rng)
        responses = np.array([
            wasserstein_distance(self.predictions, self._perturb_feature(m)) for m in masks
        ])
        surrogate = fit_weighted_ridge_surrogate(masks, responses, self.feature_columns, seed=seed)
        return BranchResult("feature", surrogate, n_perturb, seed)

    # ---- Geo branch -------------------------------------------------------

    def _geo_players(self) -> tuple[list, np.ndarray]:
        """Return (player ids, row->player index) for the Geo branch."""
        if self.spatial_unit_id is not None:
            unit = self.X[self.spatial_unit_id].astype(str).to_numpy()
            players = sorted(set(unit.tolist()))
            player_index = {p: i for i, p in enumerate(players)}
            row_to_player = np.array([player_index[u] for u in unit])
            return players, row_to_player
        players = list(range(len(self.X)))  # each row is its own spatial player
        return players, np.arange(len(self.X))

    def explain_geo_branch(
        self,
        n_perturb: int = N_PERTURB_DEFAULT,
        min_sel: int = 5,
        max_sel: int = 10,
        seed: int | None = None,
    ) -> BranchResult:
        seed = self.seed if seed is None else seed
        players, row_to_player = self._geo_players()
        n_players = len(players)
        max_sel = min(max_sel, n_players)
        min_sel = min(min_sel, max_sel)

        rng = np.random.default_rng(seed)
        masks = np.zeros((n_perturb, n_players), dtype=np.int8)
        responses = np.zeros(n_perturb, dtype=float)
        for j in range(n_perturb):
            n_sel = int(rng.integers(min_sel, max_sel + 1))
            sel = rng.choice(n_players, size=n_sel, replace=False)
            masks[j, sel] = 1
            row_mask = np.isin(row_to_player, sel)
            perturbed = self.X[self.all_columns].copy()
            perturbed.loc[row_mask, self.spatial_columns] = self.baseline[self.spatial_columns].to_numpy()
            pred = np.asarray(self.model.predict(perturbed), float)
            responses[j] = wasserstein_distance(self.predictions[row_mask], pred[row_mask]) if row_mask.any() else 0.0

        surrogate = fit_weighted_ridge_surrogate(masks, responses, [str(p) for p in players], seed=seed)
        return BranchResult("geo", surrogate, n_perturb, seed)

    def explain(self, n_perturb: int = N_PERTURB_DEFAULT, seed: int | None = None) -> dict[str, BranchResult]:
        return {
            "feature": self.explain_feature_branch(n_perturb, seed),
            "geo": self.explain_geo_branch(n_perturb, seed=seed),
        }

    # ----------------------------------------------------------------------
    # Evaluation: ATT Accuracy
    # ----------------------------------------------------------------------

    @staticmethod
    def accuracy(result: BranchResult, reference_players: set[str], k: int | None = None) -> dict[str, float]:
        """ATT Accuracy against an external reference set of important players."""
        imp = result.importance.copy()
        imp["player"] = imp["player"].astype(str)
        k = k or len(reference_players)
        y_ref = imp["player"].isin(reference_players).to_numpy(dtype=int)
        predicted_top = set(imp.sort_values("abs_coefficient", ascending=False).head(k)["player"])
        y_pred = imp["player"].isin(predicted_top).to_numpy(dtype=int)

        tp = int(np.sum((y_pred == 1) & (y_ref == 1)))
        tn = int(np.sum((y_pred == 0) & (y_ref == 0)))
        fp = int(np.sum((y_pred == 1) & (y_ref == 0)))
        fn = int(np.sum((y_pred == 0) & (y_ref == 1)))
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        try:
            auroc = float(roc_auc_score(y_ref, imp["abs_coefficient"]))
        except ValueError:
            auroc = float("nan")
        union = predicted_top | reference_players
        jaccard = len(predicted_top & reference_players) / len(union) if union else float("nan")
        return {
            "ATT_ACC": (tp + tn) / (tp + tn + fp + fn),
            "ATT_F1": f1,
            "ATT_AUROC": auroc,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "jaccard_at_k": jaccard,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        }

    # ----------------------------------------------------------------------
    # Evaluation: ATT Faithfulness
    # ----------------------------------------------------------------------

    def faithfulness(
        self,
        result: BranchResult,
        k: int,
        n_random: int = 1000,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Top-k vs. bottom-k vs. random-k intervention test (model-based,
        does not require an external reference)."""
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        imp = result.importance
        ranked = imp.sort_values("abs_coefficient", ascending=False)["player"].tolist()
        top_k, bottom_k = ranked[:k], ranked[-k:]

        if result.branch == "feature":
            respond = self._intervene_feature
        else:
            respond = self._intervene_geo

        w_top = respond(top_k)
        w_bottom = respond(bottom_k)
        all_players = imp["player"].tolist()
        random_responses = np.array([
            respond(list(rng.choice(all_players, size=k, replace=False))) for _ in range(n_random)
        ])
        random_mean = float(random_responses.mean())
        ci_lo, ci_hi = np.percentile(random_responses, [2.5, 97.5])
        p_emp = (1 + int(np.sum(random_responses >= w_top))) / (n_random + 1)
        return {
            "top_response": w_top,
            "bottom_response": w_bottom,
            "random_mean": random_mean,
            "random_ci95": (float(ci_lo), float(ci_hi)),
            "top_minus_random": w_top - random_mean,
            "top_minus_bottom": w_top - w_bottom,
            "empirical_p": p_emp,
            "n_random": n_random,
        }

    def _intervene_feature(self, players: list[str]) -> float:
        perturbed = self.X[self.all_columns].copy()
        for col in players:
            perturbed[col] = self.baseline[col]
        pred = np.asarray(self.model.predict(perturbed), float)
        return float(wasserstein_distance(self.predictions, pred))

    def _intervene_geo(self, players: list[str]) -> float:
        all_players, row_to_player = self._geo_players()
        idx = {str(p): i for i, p in enumerate(all_players)}
        sel = [idx[p] for p in players if p in idx]
        row_mask = np.isin(row_to_player, sel)
        perturbed = self.X[self.all_columns].copy()
        perturbed.loc[row_mask, self.spatial_columns] = self.baseline[self.spatial_columns].to_numpy()
        pred = np.asarray(self.model.predict(perturbed), float)
        return float(wasserstein_distance(self.predictions[row_mask], pred[row_mask])) if row_mask.any() else 0.0

    # ----------------------------------------------------------------------
    # Evaluation: ATT Stability (controlled, user-supplied modification)
    # ----------------------------------------------------------------------

    def stability(
        self,
        baseline_result: BranchResult,
        modified_geosmile: "GeoSMILE",
        k: int,
        n_perturb: int | None = None,
    ) -> dict[str, float]:
        """Compare `baseline_result` against the same branch recomputed on a
        second GeoSMILE instance built from a controlled modification of the
        data (e.g. one added spatial unit). Both instances should share the
        fitted model, surrogate spec, seed, and perturbation budget; only the
        input/reference should differ.
        """
        n_perturb = n_perturb or baseline_result.n_perturb_requested
        modified_result = (
            modified_geosmile.explain_feature_branch(n_perturb, seed=baseline_result.seed)
            if baseline_result.branch == "feature"
            else modified_geosmile.explain_geo_branch(n_perturb, seed=baseline_result.seed)
        )

        base_top = set(baseline_result.importance.sort_values("abs_coefficient", ascending=False).head(k)["player"])
        mod_top = set(modified_result.importance.sort_values("abs_coefficient", ascending=False).head(k)["player"])
        union = base_top | mod_top
        jaccard = len(base_top & mod_top) / len(union) if union else float("nan")

        common = set(baseline_result.importance["player"]) & set(modified_result.importance["player"])
        b = baseline_result.importance.set_index("player").loc[list(common), "abs_coefficient"]
        m = modified_result.importance.set_index("player").loc[list(common), "abs_coefficient"]
        rho = float(spearmanr(b, m).correlation) if len(common) > 1 else float("nan")

        pred_rho = float(spearmanr(self.predictions, modified_geosmile.predictions[: len(self.predictions)]).correlation)
        return {"stability_jaccard": jaccard, "stability_spearman": rho, "prediction_spearman_control": pred_rho}

    # ----------------------------------------------------------------------
    # Evaluation: ATT Consistency
    # ----------------------------------------------------------------------

    def consistency(
        self,
        branch: Literal["feature", "geo"],
        seeds: Sequence[int],
        n_perturb: int = N_PERTURB_DEFAULT,
        top_pct: float = 20.0,
    ) -> dict[str, object]:
        """Reproducibility across independently seeded perturbation runs.

        Primary measure (Dehghani-style): across-run coefficient variance
        and SD, per player, summarised by mean/median/max. Supplementary:
        mean pairwise rank Spearman, score Pearson, and top-`top_pct`%
        Jaccard across all run pairs.
        """
        explain = self.explain_feature_branch if branch == "feature" else self.explain_geo_branch
        runs = [explain(n_perturb, seed=s) for s in seeds]
        players = runs[0].importance.sort_values("player")["player"].tolist()
        coef_matrix = np.array([
            r.importance.set_index("player").loc[players, "coefficient"].to_numpy() for r in runs
        ])  # (R, P)

        variance = coef_matrix.var(axis=0, ddof=1)
        sd = coef_matrix.std(axis=0, ddof=1)
        per_player = pd.DataFrame({"player": players, "mean_coefficient": coef_matrix.mean(0),
                                    "variance": variance, "sd": sd})

        rank_vals, score_vals, jac_vals = [], [], []
        abs_matrix = np.abs(coef_matrix)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = abs_matrix[i], abs_matrix[j]
                rank_vals.append(spearmanr(a, b).correlation)
                score_vals.append(pearsonr(a, b)[0])
                ta, tb = np.percentile(a, 100 - top_pct), np.percentile(b, 100 - top_pct)
                A, B = set(np.flatnonzero(a >= ta)), set(np.flatnonzero(b >= tb))
                jac_vals.append(len(A & B) / len(A | B) if (A | B) else 1.0)

        return {
            "seeds": list(seeds),
            "per_player": per_player,
            "variance_mean": float(variance.mean()), "variance_median": float(np.median(variance)),
            "sd_mean": float(sd.mean()), "sd_median": float(np.median(sd)), "sd_max": float(sd.max()),
            "rank_spearman_mean": float(np.nanmean(rank_vals)),
            "score_pearson_mean": float(np.nanmean(score_vals)),
            f"top_{int(top_pct)}pct_jaccard_mean": float(np.nanmean(jac_vals)),
        }


# --------------------------------------------------------------------------
# Runnable demo on synthetic data (proves the code works on ANY dataset)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    n_rows, n_zones = 600, 40
    zone_id = rng.integers(0, n_zones, size=n_rows)
    zone_easting = rng.uniform(0, 100, size=n_zones)[zone_id]
    zone_northing = rng.uniform(0, 100, size=n_zones)[zone_id]
    demo = pd.DataFrame({
        "zone_id": zone_id,
        "easting": zone_easting + rng.normal(0, 0.5, n_rows),
        "northing": zone_northing + rng.normal(0, 0.5, n_rows),
        "feat_a": rng.normal(size=n_rows),
        "feat_b": rng.normal(size=n_rows),
        "feat_c": rng.normal(size=n_rows),
    })
    y = (
        3.0 * demo["feat_a"]
        - 1.5 * demo["feat_b"]
        + 0.05 * (demo["easting"] - 50) ** 2 / 25
        + rng.normal(scale=0.5, size=n_rows)
    )

    explainer = GeoSMILE(
        X=demo, y=y,
        feature_columns=["feat_a", "feat_b", "feat_c"],
        spatial_columns=["easting", "northing"],
        spatial_unit_id="zone_id",
        estimator=RandomForestRegressor(n_estimators=200, random_state=42),
        seed=42,
    )

    feat = explainer.explain_feature_branch(n_perturb=500)  # small N for a fast demo
    geo = explainer.explain_geo_branch(n_perturb=500)

    print("Feature importance:\n", feat.importance, "\n")
    print("Feature fidelity:", feat.fidelity, "\n")
    print("Geo importance (top 5):\n", geo.importance.head(), "\n")
    print("Geo fidelity:", geo.fidelity, "\n")

    faith = explainer.faithfulness(feat, k=2, n_random=200)
    print("Feature faithfulness:", faith, "\n")

    cons = explainer.consistency("feature", seeds=[1, 2, 3], n_perturb=300)
    print("Feature consistency (mean SD):", cons["sd_mean"], "rank rho:", cons["rank_spearman_mean"])
