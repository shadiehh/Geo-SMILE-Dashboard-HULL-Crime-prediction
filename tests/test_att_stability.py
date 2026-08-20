"""Tests for cardinal-neighbor ATT stability helpers."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from compute_att_stability import _jaccard, compare_attributions
from hull_boundaries import select_cardinal_neighbors
from hull_boundary_neighbors import expand_panel


def _square(x: float, y: float, code: str, name: str) -> dict:
    return {
        "lsoa21cd": code,
        "lsoa21nm": name,
        "geometry": box(x - 0.4, y - 0.4, x + 0.4, y + 0.4),
    }


def test_select_cardinal_neighbors_picks_one_per_direction():
    hull = gpd.GeoDataFrame(
        [_square(0.0, 0.0, "H1", "Hull 001A")],
        crs="EPSG:27700",
    )
    candidates = gpd.GeoDataFrame(
        [
            _square(0.0, 0.0, "H1", "Hull 001A"),
            _square(0.0, 2.0, "N1", "North 001A"),
            _square(2.0, 0.0, "E1", "East 001A"),
            _square(0.0, -2.0, "S1", "South 001A"),
            _square(-2.0, 0.0, "W1", "West 001A"),
            _square(4.0, 4.0, "NE1", "Far NE 001A"),
        ],
        crs="EPSG:27700",
    )
    picked = select_cardinal_neighbors(hull, candidates, max_gap_m=10.0)
    by_dir = dict(zip(picked["direction"].astype(str), picked["lsoa21cd"].astype(str)))
    assert by_dir == {"north": "N1", "east": "E1", "south": "S1", "west": "W1"}
    assert len(set(by_dir.values())) == 4


def test_select_cardinal_neighbors_honours_preferred_codes():
    from hull_boundaries import PREFERRED_CARDINAL_LSOAS

    hull = gpd.GeoDataFrame(
        [_square(0.0, 0.0, "H1", "Hull 001A")],
        crs="EPSG:27700",
    )
    north_code = PREFERRED_CARDINAL_LSOAS["north"]
    south_code = PREFERRED_CARDINAL_LSOAS["south"]
    candidates = gpd.GeoDataFrame(
        [
            _square(0.0, 0.0, "H1", "Hull 001A"),
            _square(0.0, 3.0, "N_BIG", "North big"),
            _square(0.0, 2.0, north_code, "North preferred"),
            _square(2.0, 0.0, "E1", "East 001A"),
            _square(0.0, -3.0, "S_GOLF", "South golf"),
            _square(0.0, -2.0, south_code, "South beyond golf"),
            _square(-2.0, 0.0, "W1", "West 001A"),
        ],
        crs="EPSG:27700",
    )
    picked = select_cardinal_neighbors(hull, candidates, max_gap_m=10.0)
    by_dir = dict(zip(picked["direction"].astype(str), picked["lsoa21cd"].astype(str)))
    assert by_dir["north"] == north_code
    assert by_dir["south"] == south_code
    assert by_dir["east"] == "E1"
    assert by_dir["west"] == "W1"


def test_jaccard_and_compare_attributions():
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    base_feat = pd.DataFrame(
        {"feature": ["x", "y", "z"], "abs_coefficient": [3.0, 2.0, 1.0]}
    )
    pert_feat = pd.DataFrame(
        {"feature": ["x", "y", "z"], "abs_coefficient": [2.5, 2.4, 0.1]}
    )
    base_geo = pd.DataFrame(
        {"lsoa21cd": ["A", "B", "C"], "geo_abs_coefficient": [9.0, 4.0, 1.0]}
    )
    pert_geo = pd.DataFrame(
        {"lsoa21cd": ["A", "B", "C", "N"], "geo_abs_coefficient": [8.0, 5.0, 0.5, 7.0]}
    )
    out = compare_attributions(
        base_feat,
        pert_feat,
        base_geo,
        pert_geo,
        base_feat_top={"x", "y"},
        base_geo_top={"A", "B"},
    )
    assert 0.0 <= out["feature_jaccard"] <= 1.0
    assert 0.0 <= out["geo_jaccard"] <= 1.0
    assert out["feature_spearman"] > 0.5


def test_expand_panel_appends_neighbor_rows():
    hull = pd.DataFrame(
        {
            "lsoa21cd": ["H1", "H1"],
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "crime_count": [1, 2],
            "centroid_easting": [1.0, 1.0],
            "centroid_northing": [2.0, 2.0],
        }
    )
    extra = pd.DataFrame(
        {
            "lsoa21cd": ["N1", "N1"],
            "month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "crime_count": [3, 4],
            "centroid_easting": [9.0, 9.0],
            "centroid_northing": [8.0, 8.0],
        }
    )
    out = expand_panel(hull, extra)
    assert set(out["lsoa21cd"]) == {"H1", "N1"}
    assert len(out) == 4
    assert list(out.columns) == list(hull.columns)


def test_geo_smile_extra_player_keeps_core_sample():
    import numpy as np
    from geosmile_hull_twobranch import SPATIAL_COLUMNS, geo_smile_coefficients

    rows = []
    for code, east, north in (("A", 1.0, 1.0), ("B", 2.0, 2.0), ("X", 9.0, 9.0)):
        for month in pd.date_range("2020-01-01", periods=4, freq="MS"):
            rows.append(
                {
                    "lsoa21cd": code,
                    "lsoa21nm": code,
                    "month": month,
                    "centroid_easting": east,
                    "centroid_northing": north,
                }
            )
    panel = pd.DataFrame(rows)
    mask = pd.Series(True, index=panel.index)
    baselines = panel[SPATIAL_COLUMNS].median()

    class _Stub:
        def predict(self, x):
            return np.asarray(x, dtype=float).sum(axis=1)

    core, diag_core = geo_smile_coefficients(
        _Stub(),
        panel[panel["lsoa21cd"].isin(["A", "B"])],
        pd.Series(True, index=panel[panel["lsoa21cd"].isin(["A", "B"])].index),
        baselines,
        40,
        42,
        min_sel=1,
        max_sel=2,
        sample_lsoas=["A", "B"],
    )
    expanded, diag = geo_smile_coefficients(
        _Stub(),
        panel,
        mask,
        baselines,
        40,
        42,
        min_sel=1,
        max_sel=2,
        sample_lsoas=["A", "B"],
    )
    assert diag["n_sampled_lsoas"] == 2
    assert diag["n_extra_lsoas"] == 1
    assert "X" in set(expanded["lsoa21cd"].astype(str))
    extra = expanded.loc[expanded["lsoa21cd"].astype(str) == "X"].iloc[0]
    assert float(extra["geo_abs_coefficient"]) >= 0.0
    merged = core.merge(expanded, on="lsoa21cd", suffixes=("_core", "_exp"))
    assert set(merged["lsoa21cd"]) == {"A", "B"}


def test_stability_uses_published_smile_design():
    from compute_att_stability import FEAT_MASKS, GEO_PERTURBS, SEED

    assert FEAT_MASKS == 1024
    assert GEO_PERTURBS == 2000
    assert SEED == 42
