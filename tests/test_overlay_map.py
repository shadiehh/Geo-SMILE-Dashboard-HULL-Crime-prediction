"""Tests for the GeoSensitivity / latitude / longitude overlay map."""

from __future__ import annotations

import numpy as np

from map_lat_lon_geo_overlay import load_overlay_table, overlay_channels


def test_independent_scaling_does_not_let_geo_drown_lat_lon():
    geo = np.array([24.0, 0.2, 0.2])
    lat = np.array([0.02, 0.4, 0.02])
    lon = np.array([0.02, 0.02, 0.5])
    out = overlay_channels(geo, lat, lon, pct=100)
    assert out["red"][0] == 1.0
    assert out["green"][1] == 1.0
    assert out["blue"][2] == 1.0
    assert out["dominant"][0] == "geo_sensitivity"
    assert out["dominant"][1] == "latitude_effect"
    assert out["dominant"][2] == "longitude_effect"


def test_overlay_table_has_168_lsoas_and_rgb_in_unit_interval():
    table = load_overlay_table()
    assert len(table) == 168
    assert table["lsoa21cd"].is_unique
    for col in ("overlay_r", "overlay_g", "overlay_b", "geo_unit", "lat_unit", "lon_unit"):
        assert table[col].between(0.0, 1.0).all()
    assert set(table["dominant_layer"]).issubset(
        {"geo_sensitivity", "latitude_effect", "longitude_effect", "tie", "none"}
    )
    assert not np.allclose(
        table["geo_coefficient"].to_numpy(), table["centroid_latitude"].to_numpy()
    )
    assert not np.allclose(
        table["centroid_latitude"].to_numpy(), table["centroid_longitude"].to_numpy()
    )


def test_top50_importance_requires_geosensitivity():
    table = load_overlay_table()
    assert int(table["in_top50_geo"].sum()) == 50
    assert int(table["in_top50_lat"].sum()) == 50
    assert int(table["in_top50_lon"].sum()) == 50
    shaded = table[table["geo_rank_shade"].notna()]
    assert len(shaded) == 50
    assert shaded["in_top50_geo"].all()
    lat_only = table[table["in_top50_lat"] & ~table["in_top50_geo"]]
    lon_only = table[table["in_top50_lon"] & ~table["in_top50_geo"]]
    assert lat_only["geo_rank_shade"].isna().all()
    assert lon_only["geo_rank_shade"].isna().all()
    assert table.loc[table["overlay_lat_on_geo"], "in_top50_geo"].all()
    assert table.loc[table["overlay_lon_on_geo"], "in_top50_geo"].all()
    assert table.loc[table["geo_rank_abs"] == 1, "in_top50_geo"].iloc[0]
