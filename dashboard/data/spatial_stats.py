"""Generic Local Moran's I / LISA clustering for any per-LSOA value.

Same Queen-contiguity method as compute_geo_spatial_patterns.py (the
published GeoSensitivity LISA), applied here to an arbitrary numeric column
so the dashboard can test whether other attributions (e.g. a global
feature's local coefficient) cluster spatially too, not only GeoSensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUAD_TO_LISA = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
P_THRESHOLD = 0.05


def build_queen_weights(geometry, id_col: str):
    from libpysal.weights import Queen, w_subset

    w = Queen.from_dataframe(geometry, ids=geometry[id_col].astype(str).tolist(), silence_warnings=True)
    islands = [str(i) for i in (w.islands or [])]
    if islands:
        keep = [i for i in w.id_order if str(i) not in set(islands)]
        w = w_subset(w, keep)
    w.transform = "r"
    return w, islands


def local_moran(
    values_by_id: dict[str, float], w, *, permutations: int = 249, seed: int = 42
) -> tuple[pd.DataFrame, float, float]:
    """Global + local Moran's I / LISA labels for one variable, aligned to w.id_order."""
    from esda.moran import Moran, Moran_Local

    order = [str(i) for i in w.id_order]
    y = np.array([values_by_id[i] for i in order], dtype=float)
    np.random.seed(seed)
    global_m = Moran(y, w, permutations=permutations)
    local_m = Moran_Local(y, w, permutations=permutations, seed=seed, n_jobs=1, alternative="directed")
    sig = np.asarray(local_m.p_sim, float) < P_THRESHOLD
    labels = np.array(["NS"] * len(order), dtype=object)
    for code, name in QUAD_TO_LISA.items():
        labels[sig & (np.asarray(local_m.q, int) == code)] = name
    frame = pd.DataFrame(
        {
            "lsoa21cd": order,
            "value": y,
            "lisa": labels,
            "local_I": local_m.Is,
            "p_value": local_m.p_sim,
        }
    )
    return frame, float(global_m.I), float(global_m.p_sim)
