"""Tests for Feature SMILE with longitude/latitude as attribution players."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geosmile_hull_feature_with_location import (
    LOCATION_PLAYERS,
    add_lonlat,
    bng_to_lonlat,
    choose_feature_players_with_location,
    local_tables,
)
from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players


def test_bng_to_lonlat_hull_is_sensible():
    lon, lat = bng_to_lonlat(pd.Series([509000.0]), pd.Series([429000.0]))
    assert -1.0 < float(lon[0]) < 0.2
    assert 53.5 < float(lat[0]) < 54.0


def test_choose_players_adds_lon_lat_to_thirty():
    panel = pd.read_csv("outputs/hull_lsoa_month_panel_with_covariates.csv", nrows=200)
    panel = add_lonlat(panel)
    base = choose_feature_players(panel.drop(columns=list(LOCATION_PLAYERS)))
    players = choose_feature_players_with_location(panel)
    assert len(base) == 30
    assert len(players) == 32
    assert players[-2:] == LOCATION_PLAYERS
    assert "centroid_longitude" in players
    assert "centroid_latitude" in players
    assert not any(c in players for c in SPATIAL_COLUMNS)
    assert panel[LOCATION_PLAYERS].notna().all().all()
    assert np.isfinite(panel["centroid_longitude"]).all()


def test_published_feature_still_forbids_location():
    panel = pd.read_csv("outputs/hull_lsoa_month_panel_with_covariates.csv", nrows=50)
    panel = add_lonlat(panel)
    try:
        choose_feature_players(panel)
    except ValueError as exc:
        assert "spatial" in str(exc).lower() or "forbidden" in str(exc).lower()
    else:
        raise AssertionError("Published Feature must reject lon/lat columns")


def test_local_tables_one_row_per_lsoa():
    local = pd.DataFrame(
        {
            "lsoa21cd": ["E1", "E2"],
            "lsoa21nm": ["A", "B"],
            "crime_lag_1": [0.10, -0.20],
            "centroid_longitude": [0.30, -0.10],
            "centroid_latitude": [0.00, 0.40],
        }
    )
    players = ["crime_lag_1", "centroid_longitude", "centroid_latitude"]
    wide_abs, long, loc = local_tables(local, players)
    assert len(loc) == 2
    assert list(loc["lsoa21cd"]) == ["E1", "E2"]
    assert loc["centroid_longitude_abs"].tolist() == [0.30, 0.10]
    assert len(long) == 6
    assert set(long["feature"]) == set(players)
    assert long.loc[long["is_location"] == 1, "feature"].nunique() == 2
    assert wide_abs["crime_lag_1"].tolist() == [0.10, 0.20]


def test_existing_local_csv_covers_all_hull_lsoas():
    path = "outputs/feature_with_location/feature_with_location_local_coefficients.csv"
    local = pd.read_csv(path)
    assert local["lsoa21cd"].is_unique
    assert len(local) == 168
    assert "centroid_longitude" in local.columns
    assert "centroid_latitude" in local.columns
    assert local[LOCATION_PLAYERS].notna().all().all()
