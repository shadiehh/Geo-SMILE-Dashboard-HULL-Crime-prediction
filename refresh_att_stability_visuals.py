"""Regenerate ATT-stability PNGs from already-saved CSVs (no SMILE recompute)."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

from compute_att_stability import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    VIZ_DIR,
    save_feature_importance_ranking,
    save_geo_rank_map,
)
from hull_boundaries import load_boundaries


def main() -> None:
    base_geo = pd.read_csv(VIZ_DIR / "baseline" / "geo_coefficients.csv")
    boundaries, _ = load_boundaries(sorted(base_geo["lsoa21cd"].astype(str)))
    if "lsoa21nm" in base_geo.columns:
        name_lookup = dict(
            zip(base_geo["lsoa21cd"].astype(str), base_geo["lsoa21nm"].astype(str))
        )
    else:
        name_lookup = {c: c for c in base_geo["lsoa21cd"].astype(str)}
    feat_xmax = float(
        pd.read_csv(VIZ_DIR / "baseline" / "feature_importance.csv")[
            "abs_coefficient"
        ].max()
    )

    for folder in sorted(p for p in VIZ_DIR.iterdir() if p.is_dir()):
        feat_path = folder / "feature_importance.csv"
        geo_path = folder / "geo_coefficients.csv"
        if not feat_path.exists() or not geo_path.exists():
            continue
        feat = pd.read_csv(feat_path)
        geo = pd.read_csv(geo_path)
        meta: dict = {}
        meta_path = folder / "snapshot_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if folder.name == "baseline":
            label = "Baseline (unperturbed)"
        elif folder.name == "all_four":
            label = "All four cardinal neighbors"
        else:
            added = meta.get("added_lsoa", "")
            label = f"{folder.name.title()} neighbor {added}".strip()

        feat_bits = [label]
        if "feature_jaccard" in meta:
            feat_bits.append(
                f"top-{FEATURE_TOP_K} Jaccard vs baseline={meta['feature_jaccard']:.3f}"
            )
        if "added_lsoa" in meta:
            feat_bits.append(f"added {meta.get('direction')} {meta['added_lsoa']}")
        feat_bits.append("Not causal.")

        geo_bits = [label]
        if "geo_jaccard" in meta:
            geo_bits.append(
                f"top-{GEO_TOP_K} Jaccard vs baseline={meta['geo_jaccard']:.3f}"
            )
        if "added_lsoa" in meta:
            geo_bits.append(f"added {meta.get('direction')} {meta['added_lsoa']}")
        if "added_name" in meta:
            geo_bits.append(meta["added_name"])

        extra_codes = [
            str(c)
            for c in geo["lsoa21cd"].astype(str)
            if str(c) not in set(boundaries["lsoa21cd"].astype(str))
        ]
        map_bounds = boundaries
        if extra_codes:
            extra = geo[geo["lsoa21cd"].astype(str).isin(extra_codes)]
            # Geometry is not in the coefficient CSV; keep Hull polygons only
            # unless a neighbor geojson sits next to the snapshot.
            neighbor_path = folder.parent.parent.parent / "data" / "cardinal_neighbor_lsoas.geojson"
            if neighbor_path.exists():
                neigh = gpd.read_file(neighbor_path)
                if neigh.crs is None:
                    neigh = neigh.set_crs("EPSG:27700")
                else:
                    neigh = neigh.to_crs("EPSG:27700")
                map_bounds = gpd.GeoDataFrame(
                    pd.concat(
                        [boundaries, neigh[neigh["lsoa21cd"].astype(str).isin(extra_codes)]],
                        ignore_index=True,
                    ),
                    crs=boundaries.crs,
                )
            name_lookup.update(
                {
                    str(r.lsoa21cd): str(r.lsoa21nm)
                    for r in extra.itertuples()
                    if "lsoa21nm" in extra.columns
                }
            )

        save_feature_importance_ranking(
            feat,
            folder / "feature_importance_ranking.png",
            title=f"Feature SMILE importance ranking\n{label}",
            footer=". ".join(feat_bits),
            xmax=feat_xmax,
        )
        added = [str(meta["added_lsoa"])] if meta.get("added_lsoa") else extra_codes
        save_geo_rank_map(
            geo,
            map_bounds,
            folder / f"geo_top{GEO_TOP_K}_map.png",
            title=f"Geo SMILE top-{GEO_TOP_K} by |coefficient|\n{label}",
            footer=". ".join(str(b) for b in geo_bits) + ".",
            name_lookup=name_lookup,
            added_codes=added,
        )
        print(f"refreshed {folder.name}")


if __name__ == "__main__":
    main()
