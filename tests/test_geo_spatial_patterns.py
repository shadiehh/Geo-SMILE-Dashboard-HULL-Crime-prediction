"""Tests for Global Moran / LISA geosensitivity patterns."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from compute_geo_spatial_patterns import (
    VALUE_FIELD,
    contiguous_clusters,
    interpret_global,
    lisa_labels,
    load_geo_frame,
    queen_weights,
    run_moran_lisa,
    sensitivity_tertiles,
)


def _grid(n: int, values: np.ndarray, cell: float = 1.0) -> gpd.GeoDataFrame:
    rows = []
    k = 0
    for r in range(n):
        for c in range(n):
            x0, y0 = c * cell, (n - 1 - r) * cell
            rows.append(
                {
                    "lsoa21cd": f"C{k:02d}",
                    "lsoa21nm": f"cell {k}",
                    "geo_coefficient": float(values[k]),
                    VALUE_FIELD: float(values[k]),
                    "geometry": box(x0, y0, x0 + cell, y0 + cell),
                }
            )
            k += 1
    return gpd.GeoDataFrame(rows, crs="EPSG:27700")


def test_interpret_global_rules():
    assert "clustering" in interpret_global(0.41, 0.001)
    assert "dispersion" in interpret_global(-0.3, 0.01)
    assert "no significant" in interpret_global(0.2, 0.4)


def test_lisa_labels_quadrants():
    q = np.array([1, 2, 3, 4, 1])
    p = np.array([0.01, 0.01, 0.01, 0.01, 0.4])
    lab = lisa_labels(q, p)
    assert list(lab) == ["HH", "LH", "LL", "HL", "NS"]


def test_contiguous_hh_cluster():
    labels = np.array(["HH", "HH", "NS", "NS"])
    ids = ["A", "B", "C", "D"]
    neighbors = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    clusters = contiguous_clusters(labels, ids, neighbors, "HH")
    assert len(clusters) == 1
    assert clusters[0]["n"] == 2
    assert clusters[0]["lsoa21cd"] == ["A", "B"]


def test_lattice_hh_block_and_hl_corner():
    """4x4: 2x2 high block is HH; isolated high corner is HL; global I > 0."""
    y = np.full(16, 1.0)
    y[[0, 1, 4, 5]] = 10.0  # 2x2 block, row-major
    y[15] = 10.0  # bottom-right corner, isolated high
    gdf = _grid(4, y)
    w, islands = queen_weights(gdf)
    assert islands == []
    order, values, global_m, local_m, labels = run_moran_lisa(gdf, w, permutations=199, seed=42)
    by_id = dict(zip(order, labels))
    assert float(global_m.I) > 0
    # Corner of the high block: all Queen neighbours are high.
    assert by_id["C00"] == "HH"
    # Isolated high corner among low neighbours.
    assert by_id["C15"] == "HL"


def test_published_geo_has_168_lsoas():
    geo = load_geo_frame()
    assert len(geo) == 168
    assert geo["lsoa21cd"].nunique() == 168
    assert geo[VALUE_FIELD].notna().all()


def test_payload_json_covers_all_published_lsoas():
    import json
    from pathlib import Path

    path = Path("outputs/spatial_patterns/llm_spatial_pattern.json")
    if not path.exists():
        import pytest

        pytest.skip("Run compute_geo_spatial_patterns.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    geo = load_geo_frame()
    json_ids = {row["lsoa21cd"] for row in payload["LOCAL_SPATIAL_PATTERN"]["lsoas"]}
    assert payload["n_lsoas"] == 168
    assert payload["n_analysed"] + len(payload["islands"]) == 168
    assert json_ids == set(geo["lsoa21cd"].astype(str))
    assert "GLOBAL_SPATIAL_PATTERN" in payload
    assert payload["LOCAL_SPATIAL_PATTERN"]["counts"]["HH"] >= 0


def test_sensitivity_tertiles_three_levels():
    labs = sensitivity_tertiles(np.arange(30, dtype=float))
    assert set(labs) <= {"LOW", "MID", "HIGH"}
    assert len(set(labs)) == 3
