"""Evidence-review workbench: filter areas, shortlist them, compare two and export the result.

Filters surface areas whose explanation evidence meets stated criteria. They rank model attributions and
recorded counts; they do not say where to act, and the exports carry that note.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.data.loader import HullBundle
from dashboard.data.schema import LISA_LABELS, LSOA_ID, human_label
from dashboard.llm import card_mode as cm
from dashboard.views.maps import predictions_for
from dashboard.views.stakeholder_evidence import stakeholder_data

NOTE = "Model attribution and recorded counts for evidence review. Not causal and not a basis for deployment or targeting."
CHANGE_OPTIONS = ("Any", "Rose by over a quarter", "Fell by over a quarter", "Within a quarter", "Not assessed")
LED_OPTIONS = ("Any", "Crime history", "Not crime history")


@st.cache_data(show_spinner=False)
def decision_table(_bundle: HullBundle) -> pd.DataFrame:
    """One row per LSOA with the published evidence the filters and comparison use."""
    d = stakeholder_data()
    lsoa = _bundle.lsoa.assign(**{LSOA_ID: _bundle.lsoa[LSOA_ID].astype(str)}).set_index(LSOA_ID)
    table = lsoa[["lsoa21nm", "locality_label", "geo_sensitivity", "geo_rank", "lisa_class"]].copy()
    table = table.join(cm.crime_signal_shares(d)).join(cm.built_share_table(d)[["built_share", "dominant_built_feature"]])
    table = table.join(cm.recorded_trend(d)[0][["previous_mean", "recent_mean", "change", "assessed"]])
    table = table.join(predictions_for(None).rename(columns={"predicted": "predicted_mean", "observed": "recorded_mean"}))
    table["area"] = table["lsoa21nm"].str.replace("Kingston upon Hull ", "", regex=False)
    table["crime_led"] = table["leading_feature"].str.startswith("crime_")
    return table.sort_values("geo_rank")


def _set_filters(**values) -> None:
    defaults = {"ds_rank": 168, "ds_lisa": [], "ds_built": 0, "ds_recent": 0, "ds_change": "Any", "ds_led": "Any"}
    st.session_state.update({**defaults, **values})


def render_filters(table: pd.DataFrame) -> set[str] | None:
    """The filter panel; returns the matching LSOAs, or None when no filter is set."""
    median = int(round(100 * float(table["recent_share"].median())))
    state = st.session_state
    active = (state.get("ds_rank", 168) != 168 or state.get("ds_lisa") or state.get("ds_built") or state.get("ds_recent")
              or state.get("ds_change", "Any") != "Any" or state.get("ds_led", "Any") != "Any")
    with st.expander("Filter areas" + (" (active)" if active else ""), expanded=bool(active)):
        st.caption("Starting points for evidence review, not recommendations.")
        presets = {
            "Top 20 location-sensitive": {"ds_rank": 20},
            "High-High location clusters": {"ds_lisa": ["HH"]},
            "Built-environment led": {"ds_built": 30},
            "Recent reliance and rising": {"ds_recent": median, "ds_change": "Rose by over a quarter"},
            "Not led by crime history": {"ds_led": "Not crime history"},
        }
        for label, values in presets.items():
            st.button(label, key=f"ds_preset_{label}", on_click=_set_filters, kwargs=values, width="stretch")
        st.button("Clear filters", key="ds_clear", on_click=_set_filters, type="tertiary")
        rank = st.slider("GeoSensitivity rank at most", 1, 168, 168, key="ds_rank")
        lisa = st.multiselect("Spatial pattern", list(table["lisa_class"].dropna().unique()),
                              format_func=lambda c: LISA_LABELS.get(c, c), key="ds_lisa")
        built = st.slider("Built-environment share at least (%)", 0, 60, 0, key="ds_built")
        recent = st.slider("Recent-month share at least (%)", 0, 60, 0, key="ds_recent")
        change = st.selectbox("Recorded change", CHANGE_OPTIONS, key="ds_change")
        led = st.selectbox("Leading local feature", LED_OPTIONS, key="ds_led")
    if rank == 168 and not lisa and not built and not recent and change == "Any" and led == "Any":
        return None
    mask = (table["geo_rank"] <= rank) & (table["built_share"] >= built / 100) & (table["recent_share"] >= recent / 100)
    if lisa:
        mask &= table["lisa_class"].isin(lisa)
    assessed = table["assessed"].astype(bool)
    if change == "Rose by over a quarter":
        mask &= assessed & (table["change"] > cm.TREND_THRESHOLD)
    elif change == "Fell by over a quarter":
        mask &= assessed & (table["change"] < -cm.TREND_THRESHOLD)
    elif change == "Within a quarter":
        mask &= assessed & (table["change"].abs() <= cm.TREND_THRESHOLD)
    elif change == "Not assessed":
        mask &= ~assessed
    if led != "Any":
        mask &= table["crime_led"] == (led == "Crime history")
    return set(table.index[mask])


def _display(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Area": table["area"], "Locality": table["locality_label"], "Geo rank": table["geo_rank"].astype(int),
        "Predicted crimes": table["predicted_mean"].round(1),
        "Recorded crimes": table["recorded_mean"].round(1),
        "GeoSensitivity": table["geo_sensitivity"].round(3),
        "Spatial pattern": table["lisa_class"].map(lambda c: LISA_LABELS.get(c, c)),
        "Leading feature": table["leading_feature"].map(human_label),
        "Crime history %": (100 * table["crime_history_share"]).round(0),
        "Recent months %": (100 * table["recent_share"]).round(0),
        "Built env. %": (100 * table["built_share"]).round(0),
        "Recorded change %": (100 * table["change"]).round(0),
    })
    return out


def open_card(lsoa: str) -> None:
    st.session_state["selected_lsoa"] = lsoa
    st.session_state["page"] = "Stakeholder Cards"


def export_buttons(bundle: HullBundle, table: pd.DataFrame, matches: set[str] | None, key: str) -> None:
    shown = table if matches is None else table[table.index.isin(matches)]
    export = shown.drop(columns=["crime_led"]).reset_index().assign(interpretation_note=NOTE)
    geometry = bundle.geometry[[LSOA_ID, "geometry"]].assign(**{LSOA_ID: bundle.geometry[LSOA_ID].astype(str)})
    st.download_button("Download CSV", export.to_csv(index=False), "geosmile_shortlist.csv", "text/csv", width="stretch",
                       key=f"{key}_csv")
    st.download_button("Download GeoJSON", geometry.merge(export, on=LSOA_ID).to_json(), "geosmile_shortlist.geojson",
                       "application/geo+json", width="stretch", key=f"{key}_geojson")


def render_shortlist(table: pd.DataFrame, matches: set[str] | None) -> None:
    shown = table if matches is None else table[table.index.isin(matches)]
    st.markdown(f"**{len(shown)} of {len(table)} areas**" + ("" if matches is not None else " (no filter set)"))
    if shown.empty:
        st.info("No area meets all of these criteria. Loosen a filter.")
        return
    event = st.dataframe(_display(shown), hide_index=True, width="stretch", height=320, key="ds_table",
                         on_select="rerun", selection_mode="single-row")
    rows = event.selection.rows if event is not None else []
    if rows:
        code = str(shown.index[rows[0]])
        if st.session_state.get("ds_table_last") != code:
            st.session_state["ds_table_last"] = code
            st.session_state["selected_lsoa"] = code
            st.rerun()
    st.caption(NOTE)


def render_compare(table: pd.DataFrame) -> None:
    d = stakeholder_data()
    codes = list(table.index)
    labels = {c: f"{table.loc[c, 'area']} · {table.loc[c, 'locality_label']}" for c in codes}
    first = st.session_state.get("selected_lsoa") or codes[0]
    left, right = st.columns(2)
    a = left.selectbox("Area A", codes, index=codes.index(first), format_func=labels.get, key=f"ds_a_{first}")
    b_default = d.default_contrast(a)
    b = right.selectbox("Area B", codes, index=codes.index(b_default), format_func=labels.get, key=f"ds_b_{a}")

    def pct(column: str, code: str | None = None) -> str:
        value = table[column].median() if code is None else table.loc[code, column]
        return "not assessed" if pd.isna(value) else f"{100 * float(value):.0f}%"

    def row(name: str, column: str) -> dict:
        return {"Measure": name, labels[a]: pct(column, a), labels[b]: pct(column, b), "Hull median": pct(column)}

    rows = [{"Measure": "GeoSensitivity rank (of 168)", labels[a]: str(int(table.loc[a, "geo_rank"])),
             labels[b]: str(int(table.loc[b, "geo_rank"])), "Hull median": "n/a"},
            {"Measure": "GeoSensitivity", labels[a]: f"{table.loc[a, 'geo_sensitivity']:.3f}",
             labels[b]: f"{table.loc[b, 'geo_sensitivity']:.3f}", "Hull median": f"{table['geo_sensitivity'].median():.3f}"},
            row("Crime history share", "crime_history_share"), row("Recent-month share", "recent_share"),
            row("Six-month average share", "long_run_share"), row("Built-environment share", "built_share"),
            row("Recorded change", "change")]
    if a == b:
        st.info("Choose two different areas.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    cols = st.columns(2)
    for col, code in zip(cols, (a, b)):
        local = d.abs_local(code)
        share = (local / local.sum()).head(5)
        col.markdown(f"**Top five features: {labels[code]}**")
        col.dataframe(pd.DataFrame({"Feature": [human_label(f) for f in share.index],
                                    "Share of local importance": [f"{100 * v:.0f}%" for v in share.values]}),
                      hide_index=True, width="stretch")
    st.caption(NOTE)


