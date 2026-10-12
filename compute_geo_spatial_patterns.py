"""Global Moran and Local Moran/LISA on published Hull GeoSMILE geosensitivity.

Value field: geo_abs_coefficient (magnitude of geographic sensitivity) by
default; --signed uses geo_coefficient (the area-data design ranks by the
signed coefficient).
Weights: Queen contiguity, row-standardised. Classical spatial statistics only.

Outputs under outputs/spatial_patterns/ are structured for another project's LLM:
the model should cite Moran I / LISA labels, not visual impressions of the map.
Not causal.

    python compute_geo_spatial_patterns.py
    python compute_geo_spatial_patterns.py --signed         --coef outputs/geo_area_data_eval/n_4000/geo_coefficients.csv         --out-dir outputs/geo_area_data_eval/spatial_patterns
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from hull_boundaries import hull_outline, load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
COEF_PATH = OUT / "geo_coefficients.csv"
PATTERN_DIR = OUT / "spatial_patterns"

VALUE_FIELD = "geo_abs_coefficient"
N_PERMUTATIONS = 999
SEED = 42
P_THRESHOLD = 0.05
WEIGHTS_LABEL = "queen_contiguity_row_standardised"

# esda.Moran_Local.q: 1=HH, 2=LH, 3=LL, 4=HL
QUAD_TO_LISA = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
LISA_ORDER = ("HH", "LL", "HL", "LH", "NS")
LISA_COLOURS = {
    "HH": "#C62828",
    "LL": "#1565C0",
    "HL": "#EF6C00",
    "LH": "#00838F",
    "NS": "#E8ECF0",
    "ISLAND": "#BDBDBD",
}
INK, INK_MUTED = "#12171C", "#57646D"


def load_geo_frame() -> pd.DataFrame:
    if not COEF_PATH.exists():
        raise SystemExit(f"Missing published Geo ranking: {COEF_PATH}")
    geo = pd.read_csv(COEF_PATH)
    if VALUE_FIELD not in geo.columns:
        geo[VALUE_FIELD] = geo["geo_coefficient"].abs()
    geo["lsoa21cd"] = geo["lsoa21cd"].astype(str)
    if "lsoa21nm" not in geo.columns:
        names = gpd.read_file(ROOT / "data" / "hull_lsoa_2021_official.geojson", ignore_geometry=True)
        geo["lsoa21nm"] = geo["lsoa21cd"].map(names.set_index("lsoa21cd")["lsoa21nm"]).fillna(geo["lsoa21cd"])
    return geo[list(dict.fromkeys(["lsoa21cd", "lsoa21nm", "geo_coefficient", VALUE_FIELD]))].copy()


def attach_boundaries(geo: pd.DataFrame) -> gpd.GeoDataFrame:
    codes = sorted(geo["lsoa21cd"].unique())
    boundaries, _ = load_boundaries(codes)
    gdf = boundaries[["lsoa21cd", "geometry"]].copy()
    gdf["lsoa21cd"] = gdf["lsoa21cd"].astype(str)
    out = gdf.merge(geo, on="lsoa21cd", how="inner", validate="one_to_one")
    missing = set(geo["lsoa21cd"]) - set(out["lsoa21cd"])
    if missing:
        raise SystemExit(f"Boundaries missing LSOAs: {sorted(missing)[:8]}")
    return out.to_crs("EPSG:27700")


def queen_weights(gdf: gpd.GeoDataFrame):
    from libpysal.weights import Queen, w_subset

    w = Queen.from_dataframe(
        gdf, ids=gdf["lsoa21cd"].astype(str).tolist(), silence_warnings=True
    )
    islands = [str(i) for i in (w.islands or [])]
    if islands:
        keep = [i for i in w.id_order if str(i) not in set(islands)]
        w = w_subset(w, keep)
    w.transform = "r"
    return w, islands


def sensitivity_tertiles(values: np.ndarray) -> np.ndarray:
    """LOW / MID / HIGH from tertiles of |coef| (rank-based to tolerate ties)."""
    ranks = pd.Series(values).rank(method="average")
    labels = pd.qcut(ranks, 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")
    return np.asarray(labels.astype(str))


def interpret_global(i_value: float, p_value: float) -> str:
    if p_value < P_THRESHOLD and i_value > 0:
        return "positive spatial clustering of geosensitivity"
    if p_value < P_THRESHOLD and i_value < 0:
        return "spatial dispersion of geosensitivity"
    return "no significant global spatial pattern"


def lisa_labels(q: np.ndarray, p_sim: np.ndarray, *, p_threshold: float = P_THRESHOLD) -> np.ndarray:
    out = np.array(["NS"] * len(q), dtype=object)
    sig = np.asarray(p_sim, float) < p_threshold
    for code, name in QUAD_TO_LISA.items():
        out[sig & (np.asarray(q, int) == code)] = name
    return out


def contiguous_clusters(
    labels: np.ndarray,
    ids: list[str],
    neighbors: dict,
    target: str,
) -> list[dict]:
    """Connected components of significant same-quadrant units on the Queen graph."""
    id_to_i = {str(c): i for i, c in enumerate(ids)}
    members = {str(ids[i]) for i, lab in enumerate(labels) if lab == target}
    seen: set[str] = set()
    clusters: list[dict] = []
    for start in sorted(members):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        group: list[str] = []
        while queue:
            node = queue.popleft()
            group.append(node)
            for nb in neighbors.get(node, []):
                nb = str(nb)
                if nb in members and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        group = sorted(group)
        clusters.append({"cluster_id": len(clusters) + 1, "n": len(group), "lsoa21cd": group})
    clusters.sort(key=lambda c: (-c["n"], c["lsoa21cd"][0]))
    for i, c in enumerate(clusters, start=1):
        c["cluster_id"] = i
    return clusters


def run_moran_lisa(gdf: gpd.GeoDataFrame, w, *, permutations: int = N_PERMUTATIONS, seed: int = SEED):
    from esda.moran import Moran, Moran_Local

    order = [str(i) for i in w.id_order]
    lookup = gdf.set_index("lsoa21cd")
    y = lookup.loc[order, VALUE_FIELD].to_numpy(dtype=float)
    np.random.seed(seed)
    global_m = Moran(y, w, permutations=permutations)
    local_m = Moran_Local(
        y,
        w,
        permutations=permutations,
        seed=seed,
        n_jobs=1,
        alternative="directed",
    )
    labels = lisa_labels(local_m.q, local_m.p_sim)
    return order, y, global_m, local_m, labels


def build_payload(
    geo: pd.DataFrame,
    analysed: pd.DataFrame,
    global_m,
    islands: list[str],
) -> dict:
    counts = {k: int((analysed["lisa"] == k).sum()) for k in LISA_ORDER}
    counts["ISLAND"] = len(islands)
    n_total = int(len(geo))
    hh = analysed.loc[analysed["lisa"] == "HH"]
    hl_high = analysed.loc[(analysed["lisa"] == "HL") & (analysed["geo_sensitivity"] == "HIGH")]
    payload = {
        "value_field": VALUE_FIELD,
        "weights": WEIGHTS_LABEL,
        "n_lsoas": n_total,
        "n_analysed": int(len(analysed)),
        "islands": islands,
        "p_threshold": P_THRESHOLD,
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "notes": (
            f"Association of {VALUE_FIELD} in space. "
            "HH = high geosensitivity surrounded by high; HL = high surrounded by low "
            "(localised anomaly). Not a causal crime effect."
        ),
        "GLOBAL_SPATIAL_PATTERN": {
            "Moran_I": float(global_m.I),
            "expected_I": float(global_m.EI),
            "p_value": float(global_m.p_sim),
            "n_permutations": N_PERMUTATIONS,
            "interpretation": interpret_global(float(global_m.I), float(global_m.p_sim)),
        },
        "LOCAL_SPATIAL_PATTERN": {
            "method": "LISA_Moran_Local",
            "p_threshold": P_THRESHOLD,
            "counts": counts,
            "contiguous_HH_clusters": _clusters_from_frame(analysed, "HH"),
            "contiguous_LL_clusters": _clusters_from_frame(analysed, "LL"),
            "n_HIGH_HL_anomalies": int(len(hl_high)),
            "lsoas": [
                {
                    "lsoa21cd": str(r.lsoa21cd),
                    "lsoa21nm": str(r.lsoa21nm),
                    VALUE_FIELD: float(getattr(r, VALUE_FIELD)),
                    "geo_sensitivity": str(r.geo_sensitivity),
                    "lisa": str(r.lisa),
                    "local_I": float(r.local_I),
                    "p_value": float(r.p_value),
                }
                for r in analysed.sort_values(
                    [VALUE_FIELD, "lsoa21cd"], ascending=[False, True], kind="mergesort"
                ).itertuples()
            ],
        },
    }
    island_rows = geo[geo["lsoa21cd"].isin(islands)]
    for r in island_rows.itertuples():
        payload["LOCAL_SPATIAL_PATTERN"]["lsoas"].append(
            {
                "lsoa21cd": str(r.lsoa21cd),
                "lsoa21nm": str(r.lsoa21nm),
                VALUE_FIELD: float(getattr(r, VALUE_FIELD)),
                "geo_sensitivity": "NA",
                "lisa": "ISLAND",
                "local_I": None,
                "p_value": None,
            }
        )
    payload["LOCAL_SPATIAL_PATTERN"]["example_HH"] = [
        {"lsoa21cd": str(r.lsoa21cd), "lsoa21nm": str(r.lsoa21nm), VALUE_FIELD: float(getattr(r, VALUE_FIELD))}
        for r in hh.sort_values(VALUE_FIELD, ascending=False).head(8).itertuples()
    ]
    payload["LOCAL_SPATIAL_PATTERN"]["example_HIGH_HL"] = [
        {"lsoa21cd": str(r.lsoa21cd), "lsoa21nm": str(r.lsoa21nm), VALUE_FIELD: float(getattr(r, VALUE_FIELD))}
        for r in hl_high.sort_values(VALUE_FIELD, ascending=False).head(8).itertuples()
    ]
    return payload


def _clusters_from_frame(analysed: pd.DataFrame, target: str) -> list[dict]:
    neighbors = analysed.set_index("lsoa21cd")["neighbors"].to_dict()
    labels = analysed["lisa"].to_numpy()
    ids = analysed["lsoa21cd"].astype(str).tolist()
    return contiguous_clusters(labels, ids, neighbors, target)


def plot_lisa_map(gdf: gpd.GeoDataFrame, path: Path) -> None:
    plot = gdf.to_crs("EPSG:27700").copy()
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    for lab, colour in LISA_COLOURS.items():
        sub = plot[plot["lisa"] == lab]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            color=colour,
            edgecolor="white",
            linewidth=0.25,
            zorder=3 if lab != "NS" else 1,
        )
    outline = hull_outline(plot)
    if outline is not None:
        gpd.GeoSeries([outline], crs=plot.crs).plot(
            ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=5
        )
    ax.set_axis_off()
    ax.set_title(
        f"LISA of GeoSMILE geosensitivity ({VALUE_FIELD})",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    handles = [
        Patch(facecolor=LISA_COLOURS[k], edgecolor="white", label=k)
        for k in ("HH", "LL", "HL", "LH", "NS")
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        fontsize=8,
        title="LISA",
        title_fontsize=8,
    )
    fig.text(
        0.5,
        0.015,
        "HH/LL = concentrations of high/low |Geo SMILE coefficient|. "
        "HL = high unit among low neighbours (localised anomaly); LH = low among high. "
        "NS = not significant at p=0.05. Queen contiguity, 999 permutations, seed 42. "
        "Not causal. Distinct from the Geo |coefficient| choropleth.",
        ha="center",
        fontsize=6.5,
        color=INK_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    plt.close("all")
    print(f"Saved: {path}")


def write_markdown(payload: dict, path: Path) -> None:
    g = payload["GLOBAL_SPATIAL_PATTERN"]
    loc = payload["LOCAL_SPATIAL_PATTERN"]
    counts = loc["counts"]
    hh_lines = "\n".join(
        f"- cluster {c['cluster_id']}: n={c['n']}; " + ", ".join(c["lsoa21cd"])
        for c in loc["contiguous_HH_clusters"]
        if c["n"] >= 2
    ) or "- none with 2+ contiguous HH units"
    hl_lines = "\n".join(
        f"- {e['lsoa21cd']} {e.get('lsoa21nm', '')} {VALUE_FIELD}={e[VALUE_FIELD]:.3f}"
        for e in loc["example_HIGH_HL"]
    ) or "- none"
    path.write_text(
        f"""# GeoSMILE spatial pattern (LLM input)

Use these statistics. Do not infer clustering from map colour alone.

## GLOBAL_SPATIAL_PATTERN

- Moran_I: {g['Moran_I']:.4f}
- expected_I: {g['expected_I']:.4f}
- p_value: {g['p_value']:.4f}
- n_permutations: {g['n_permutations']}
- weights: {payload['weights']}
- value_field: {payload['value_field']}
- interpretation: {g['interpretation']}

Allowed claim: geosensitivity is spatially clustered if interpretation says so.
Forbidden claim: "the map appears clustered."

## LOCAL_SPATIAL_PATTERN (LISA)

Counts at p<{payload['p_threshold']}: HH={counts['HH']}, LL={counts['LL']}, HL={counts['HL']}, LH={counts['LH']}, NS={counts['NS']}.

HH = high |coefficient| surrounded by high (regional concentration).
HL = high surrounded by low (localised anomaly, not a regional pattern).

### Contiguous HH clusters (n>=2)

{hh_lines}

### HIGH geosensitivity + HL (localised anomalies)

{hl_lines}

{payload['notes']}
""",
        encoding="utf-8",
    )


def main() -> None:
    global COEF_PATH, PATTERN_DIR, VALUE_FIELD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coef", type=Path, default=COEF_PATH)
    parser.add_argument("--out-dir", type=Path, default=PATTERN_DIR)
    parser.add_argument("--signed", action="store_true",
                        help="analyse the signed geo_coefficient instead of |coefficient|")
    args = parser.parse_args()
    COEF_PATH = args.coef if args.coef.is_absolute() else ROOT / args.coef
    PATTERN_DIR = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if args.signed:
        VALUE_FIELD = "geo_coefficient"

    geo = load_geo_frame()
    gdf = attach_boundaries(geo)
    w, islands = queen_weights(gdf)
    if islands:
        print(f"Islands excluded from Moran/LISA: {islands}")
    analysed_gdf = gdf[~gdf["lsoa21cd"].isin(islands)].copy()
    order, y, global_m, local_m, labels = run_moran_lisa(analysed_gdf, w)
    analysed = pd.DataFrame(
        {
            "lsoa21cd": order,
            VALUE_FIELD: y,
            "lisa": labels,
            "local_I": local_m.Is,
            "p_value": local_m.p_sim,
            "q": local_m.q,
        }
    )
    names = analysed_gdf.set_index("lsoa21cd")["lsoa21nm"].astype(str)
    analysed["lsoa21nm"] = analysed["lsoa21cd"].map(names)
    analysed["geo_sensitivity"] = sensitivity_tertiles(analysed[VALUE_FIELD].to_numpy())
    analysed["neighbors"] = [
        [str(n) for n in w.neighbors.get(i, [])] for i in analysed["lsoa21cd"]
    ]
    payload = build_payload(geo, analysed, global_m, islands)

    PATTERN_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PATTERN_DIR / "lisa_lsoa.csv"
    csv_cols = [
        "lsoa21cd",
        "lsoa21nm",
        VALUE_FIELD,
        "geo_sensitivity",
        "lisa",
        "local_I",
        "p_value",
    ]
    analysed[csv_cols].to_csv(csv_path, index=False)

    json_path = PATTERN_DIR / "llm_spatial_pattern.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, PATTERN_DIR / "llm_spatial_pattern.md")

    map_gdf = analysed_gdf.merge(
        analysed[["lsoa21cd", "lisa", "geo_sensitivity", "local_I", "p_value"]],
        on="lsoa21cd",
        how="left",
    )
    plot_lisa_map(map_gdf, PATTERN_DIR / "lisa_map.png")

    g = payload["GLOBAL_SPATIAL_PATTERN"]
    c = payload["LOCAL_SPATIAL_PATTERN"]["counts"]
    print(
        f"Global Moran I={g['Moran_I']:.4f} p={g['p_value']:.4f} "
        f"({g['interpretation']})"
    )
    print(f"LISA counts: {c}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
