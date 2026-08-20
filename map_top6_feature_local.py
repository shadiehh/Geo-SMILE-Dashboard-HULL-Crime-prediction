"""Produce local Feature SMILE attribution maps for the top-10 global features.

Evaluation design uses Feature top-10 of 30 (not top-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geosmile_hull_twobranch import FEATURE_LABELS, plot_choropleth, shared_limit
from hull_boundaries import load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LOCAL = OUT / "feature_local_coefficients.csv"
GLOBAL = OUT / "feature_global_importance.csv"
MAP_DIR = OUT / "maps_twobranch" / "top10_feature"
TOP_K = 10


def main() -> None:
    global_imp = pd.read_csv(GLOBAL).sort_values("abs_coefficient", ascending=False)
    top = global_imp.head(TOP_K).reset_index(drop=True)
    players = top["feature"].tolist()
    print("Top-10 features by |global SMILE coefficient|:")
    for i, row in top.iterrows():
        print(f"  {i+1}. {row['feature']}  |coef|={row['abs_coefficient']:.4f}")

    local = pd.read_csv(LOCAL)
    codes = sorted(local["lsoa21cd"].unique().tolist())
    boundaries, meta = load_boundaries(codes)
    if not meta.get("valid", True):
        print("Boundary warning:", meta)

    signed_dir = MAP_DIR / "signed"
    abs_dir = MAP_DIR / "absolute"
    signed_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)

    test_period = "Jan–Apr 2026 (held-out)"
    paths = []
    panel_frames = []

    for rank, player in enumerate(players, start=1):
        label = FEATURE_LABELS.get(player, player)
        if player not in local.columns:
            raise SystemExit(f"Missing local coefficient column: {player}")
        merged = boundaries.merge(
            local[["lsoa21cd", player]], on="lsoa21cd", how="inner", validate="one_to_one"
        )
        merged["abs_value"] = merged[player].abs()
        vmax = shared_limit(merged[player].to_numpy())
        abs_vmax = shared_limit(merged["abs_value"].to_numpy())

        s_path = signed_dir / f"rank{rank:02d}_{player}_signed.png"
        a_path = abs_dir / f"rank{rank:02d}_{player}_abs.png"
        plot_choropleth(
            merged,
            player,
            f"Hull — Top-{rank} Feature: {label}\nLocal SMILE coefficient (signed)",
            s_path,
            signed=True,
            vmax=vmax,
            test_period=test_period,
            cbar_label="Signed local SMILE coefficient",
        )
        plot_choropleth(
            merged,
            "abs_value",
            f"Hull — Top-{rank} Feature: {label}\nLocal SMILE coefficient (absolute)",
            a_path,
            signed=False,
            vmax=abs_vmax,
            test_period=test_period,
            cbar_label="|Local SMILE coefficient|",
        )
        paths.append({"rank": rank, "feature": player, "signed": str(s_path), "absolute": str(a_path)})
        panel_frames.append((rank, player, label, merged, vmax))

    # Combined signed panel (2x5 for top-10)
    fig, axes = plt.subplots(2, 5, figsize=(18, 9))
    plot_gdf_base = boundaries.to_crs("EPSG:27700")
    from matplotlib.colors import TwoSlopeNorm
    from geosmile_hull_twobranch import DIVERGING, INK, INK_MUTED, hull_outline

    for ax, (rank, player, label, merged, vmax) in zip(axes.ravel(), panel_frames):
        g = plot_gdf_base.merge(merged[["lsoa21cd", player]], on="lsoa21cd")
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        g.plot(column=player, cmap=DIVERGING, norm=norm, edgecolor="white", linewidth=0.08, ax=ax)
        gpd.GeoSeries([hull_outline(g)], crs=g.crs).plot(ax=ax, color="none", edgecolor=INK, linewidth=0.8)
        ax.set_axis_off()
        ax.set_title(f"{rank}. {label}", fontsize=9, color=INK)

    fig.suptitle(
        "Top-10 Feature local SMILE attributions (signed)\nFeature branch — location excluded",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.01,
        "Local Ridge SMILE coefficients averaged to LSOA. "
        "Evaluation design: Feature top-10 of 30.",
        ha="center",
        fontsize=7,
        color=INK_MUTED,
    )
    panel_path = MAP_DIR / "top10_feature_local_signed_panel.png"
    fig.savefig(panel_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {panel_path}")

    manifest = {
        "top_k": TOP_K,
        "universe_n": 30,
        "ranking_source": str(GLOBAL),
        "local_source": str(LOCAL),
        "features": top.to_dict(orient="records"),
        "maps": paths,
        "panel": str(panel_path),
    }
    (MAP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    top.to_csv(MAP_DIR / "top10_features.csv", index=False)
    print(f"\nMaps written to {MAP_DIR.resolve()}")


if __name__ == "__main__":
    main()
