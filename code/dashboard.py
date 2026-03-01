# app.py — Streamlit dashboard for county-level overdose mortality (CDC WONDER extracts)
# Run: streamlit run app.py

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import streamlit as st
import plotly.express as px

# -----------------------------
# Paths 
# -----------------------------
DATA_PATH = Path("/Users/aleena/Desktop/GitHub Folder/pubhealth-dataviz/data")
DERIVED = DATA_PATH / "derived"
TIGER_RAW = DATA_PATH / "raw" / "spatial-county"
SHAPEFILE = TIGER_RAW / "tl_2025_us_county.shp"
STATE_RAW = DATA_PATH / "raw" / "spatial-state"
STATE_SHAPEFILE = STATE_RAW / "tl_2025_us_state.shp"  

st.set_page_config(page_title="Drug-related mortality disparity", layout="wide")

# -----------------------------
# Small helpers (to match my preprocessing.py file conventions)
# -----------------------------
def _snake_case(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("%", "pct").replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _standardize_fips(x: pd.Series) -> pd.Series:
    return (
        x.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

EXCLUDE_STATEFPS = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, AS, GU, MP, PR, VI

# -----------------------------
# Load geometry (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_counties_geojson(shapefile_path: str, simplify_tol: float = 0.01) -> tuple[dict, pd.DataFrame]:
    counties = gpd.read_file(shapefile_path)
    counties.columns = [_snake_case(c) for c in counties.columns]

    geoid_col = "geoid" if "geoid" in counties.columns else ("geoid10" if "geoid10" in counties.columns else None)
    if geoid_col is None:
        raise KeyError("County shapefile is missing GEOID/GEOID10.")

    counties["fips"] = _standardize_fips(counties[geoid_col])
    counties = counties.to_crs(epsg=4326)

    # Exclude non-contiguous + territories
    counties = counties.loc[~counties["fips"].str[:2].isin(EXCLUDE_STATEFPS)].copy()

    # ✅ Simplify geometry (biggest win)
    # tolerance is in degrees in EPSG:4326; 0.01 is a good starting point
    counties["geometry"] = counties["geometry"].simplify(simplify_tol, preserve_topology=True)

    # Keep minimal columns for browser payload
    attr_cols = [c for c in ["fips", "namelsad", "name"] if c in counties.columns]
    attr = counties[attr_cols].copy()

    counties_geo = counties[["fips", "geometry"]].copy()
    counties_geojson = counties_geo.__geo_interface__
    return counties_geojson, attr

# cached loader for states 
@st.cache_data(show_spinner=False)
def load_states_geojson(shapefile_path: str, simplify_tol: float = 0.02) -> tuple[dict, pd.DataFrame]:
    states = gpd.read_file(shapefile_path)
    states.columns = [_snake_case(c) for c in states.columns]

    # TIGER state files typically have: geoid (2-digit), statefp (2-digit), stusps, name
    if "statefp" not in states.columns:
        if "geoid" in states.columns:
            states["statefp"] = states["geoid"].astype(str).str.zfill(2)
        else:
            raise KeyError("State shapefile missing 'STATEFP' (or 'GEOID').")

    states["statefp"] = states["statefp"].astype(str).str.zfill(2)
    states = states.to_crs(epsg=4326)

    # Exclude AK, HI, territories (keep Lower 48 + DC)
    states = states.loc[~states["statefp"].isin(EXCLUDE_STATEFPS)].copy()

    # Simplify to keep payload small
    states["geometry"] = states["geometry"].simplify(simplify_tol, preserve_topology=True)

    # Minimal columns for labels + future policy joins
    attr_cols = [c for c in ["statefp", "stusps", "name"] if c in states.columns]
    attr = states[attr_cols].copy()

    states_geo = states[["statefp", "geometry"]].copy()
    states_geojson = states_geo.__geo_interface__
    return states_geojson, attr

# -----------------------------
# Load CDC derived data (cached)
# -----------------------------
DATASETS = {
    "Overall (all counties, 2018–2023)": "overall_county_2018_2023_all_mcd_types.csv",
    "Urbanization (2023 category)": "county_urbanization2023_all_mcd_types.csv",
    "Sex (county × sex)": "county_sex_2018_2023_all_mcd_types.csv",
    "Race (county × race)": "county_race_2018_2023_all_mcd_types.csv",
    "Education (county × education)": "county_education_2018_2023_all_mcd_types.csv",
}

DEFAULT_DATASET_LABEL = "Overall (all counties, 2018–2023)"  # ✅ default landing view

@st.cache_data(show_spinner=False)
def load_cdc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [_snake_case(c) for c in df.columns]
    if "fips" not in df.columns:
        raise KeyError("CDC file missing fips.")
    df["fips"] = _standardize_fips(df["fips"])
    df["statefp"] = df["fips"].str[:2]

    for c in ["year", "deaths", "population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.loc[~df["statefp"].isin(EXCLUDE_STATEFPS)].copy()
    return df

# -----------------------------
# Aggregation for mapping
# -----------------------------
def aggregate_county(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    out = (
        df.groupby(by, as_index=False)
          .agg(deaths=("deaths", "sum"), population=("population", "sum"))
    )
    out["rate_per_100k"] = (out["deaths"] / out["population"]) * 100_000.0
    out.loc[out["population"].isna() | (out["population"] <= 0), "rate_per_100k"] = np.nan

    out["value"] = out["deaths"] if metric == "burden" else out["rate_per_100k"]
    return out

# -----------------------------
# UI DESIGN
# -----------------------------
st.title("Drug-related mortality disparity: county dashboard")

with st.sidebar:
    st.header("Controls")

    dataset_label = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(DEFAULT_DATASET_LABEL),
    )
    cdc_path = str(DERIVED / DATASETS[dataset_label])

    # ✅ default landing view: overall-substances rate
    metric = st.radio("Metric", ["Rate per 100k", "Death burden"], index=0, horizontal=False)
    metric_key = "rate" if metric.startswith("Rate") else "burden"

    st.divider()
    st.subheader("Filters")

    df = load_cdc_csv(cdc_path)

    if "year" in df.columns and df["year"].notna().any():
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
        df = df.loc[df["year"].between(year_range[0], year_range[1])].copy()
    else:
        year_range = None

    if "mcd_type" in df.columns:
        mcd_types = sorted([x for x in df["mcd_type"].dropna().unique().tolist()])
        # ✅ default landing view: "All" substances
        mcd_sel = st.selectbox("Substance (mcd_type)", ["All"] + mcd_types, index=0)
        if mcd_sel != "All":
            df = df.loc[df["mcd_type"] == mcd_sel].copy()
    else:
        mcd_sel = None

    strat_cols = []
    for c in ["2023_urbanization", "sex", "single_race_6", "education"]:
        if c in df.columns:
            strat_cols.append(c)

    strat_col = None
    strat_value = None
    if len(strat_cols) > 0:
        strat_col = st.selectbox("Stratify by (optional)", ["None"] + strat_cols, index=0)
        if strat_col != "None":
            vals = sorted([x for x in df[strat_col].dropna().unique().tolist()])
            strat_value = st.selectbox(f"{strat_col}", ["All"] + vals, index=0)
            if strat_value != "All":
                df = df.loc[df[strat_col] == strat_value].copy()
        else:
            strat_col = None

    st.divider()
    top_n = st.slider("Top counties table (N)", 5, 25, 10)

# -----------------------------
# Load geometry
# -----------------------------
with st.spinner("Loading boundaries..."):
    counties_geojson, counties_attr = load_counties_geojson(str(SHAPEFILE), simplify_tol=0.01)
    states_geojson, states_attr = load_states_geojson(str(STATE_SHAPEFILE), simplify_tol=0.02)

# Build state dropdown options using USPS abbreviation if available
if "stusps" in states_attr.columns:
    state_options = ["All"] + sorted(states_attr["stusps"].dropna().unique().tolist())
    st.sidebar.empty()  # no-op; keeps Streamlit happy if you refactor later
else:
    # fallback to STATEFP
    state_options = ["All"] + sorted(states_attr["statefp"].dropna().unique().tolist())

# Re-render the selectbox with real options (same label, stable key)
with st.sidebar:
    state_filter = st.selectbox("State filter (optional)", state_options, index=0, key="state_filter")

# Convert selection to STATEFP
if state_filter != "All":
    if "stusps" in states_attr.columns:
        sel_statefp = states_attr.loc[states_attr["stusps"] == state_filter, "statefp"].iloc[0]
    else:
        sel_statefp = state_filter

    # Filter the CDC dataframe to selected state
    df = df.loc[df["statefp"] == sel_statefp].copy()

    # Filter county attributes (for names/labels)
    counties_attr = counties_attr.loc[counties_attr["fips"].str[:2] == sel_statefp].copy()

    # Filter county geojson to selected state (keeps browser payload small)
    # Easiest way: reload counties geometries as a GeoDataFrame and subset once.
    # Minimal overhead since cached; we do it only when filtering.
    # (We keep this small by reading from cached shapefile but filtering in memory.)
    # NOTE: to avoid re-reading, you can pre-build a counties_gdf cache later.
    counties_gdf = gpd.read_file(str(SHAPEFILE))
    counties_gdf.columns = [_snake_case(c) for c in counties_gdf.columns]
    geoid_col = "geoid" if "geoid" in counties_gdf.columns else ("geoid10" if "geoid10" in counties_gdf.columns else None)
    counties_gdf["fips"] = _standardize_fips(counties_gdf[geoid_col])
    counties_gdf = counties_gdf.to_crs(epsg=4326)
    counties_gdf = counties_gdf.loc[~counties_gdf["fips"].str[:2].isin(EXCLUDE_STATEFPS)].copy()
    counties_gdf = counties_gdf.loc[counties_gdf["fips"].str[:2] == sel_statefp].copy()
    counties_gdf["geometry"] = counties_gdf["geometry"].simplify(0.01, preserve_topology=True)
    counties_geojson = counties_gdf[["fips", "geometry"]].__geo_interface__

    # Filter state geojson to selected state (for the mini basemap + zoom)
    states_gdf = gpd.read_file(str(STATE_SHAPEFILE))
    states_gdf.columns = [_snake_case(c) for c in states_gdf.columns]
    if "statefp" not in states_gdf.columns:
        states_gdf["statefp"] = states_gdf["geoid"].astype(str).str.zfill(2)
    states_gdf["statefp"] = states_gdf["statefp"].astype(str).str.zfill(2)
    states_gdf = states_gdf.to_crs(epsg=4326)
    states_gdf = states_gdf.loc[~states_gdf["statefp"].isin(EXCLUDE_STATEFPS)].copy()
    states_gdf = states_gdf.loc[states_gdf["statefp"] == sel_statefp].copy()
    states_gdf["geometry"] = states_gdf["geometry"].simplify(0.02, preserve_topology=True)
    states_geojson = states_gdf[["statefp", "geometry"]].__geo_interface__

# -----------------------------
# Build mapping frame
# -----------------------------
group_keys = ["fips"]
if "mcd_type" in df.columns:
    group_keys.append("mcd_type")
if strat_col is not None:
    group_keys.append(strat_col)

agg = aggregate_county(df, by=group_keys, metric=metric_key)

# Map layer selection:
# If "All" substances, aggregate across mcd_type for map
if "mcd_type" in group_keys and (mcd_sel is None or mcd_sel == "All"):
    agg_map = aggregate_county(df, by=["fips"], metric=metric_key)
else:
    agg_map = agg.copy()
    if strat_col is not None and strat_value == "All":
        agg_map = aggregate_county(df, by=["fips"], metric=metric_key)

agg_map = agg_map.merge(counties_attr, on="fips", how="left")

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.subheader("Interactive county map")

    title_bits = [dataset_label, metric]
    if year_range is not None:
        title_bits.append(f"{year_range[0]}–{year_range[1]}")
    if mcd_sel and mcd_sel != "All":
        title_bits.append(mcd_sel)
    if strat_col and strat_value and strat_value != "All":
        title_bits.append(f"{strat_col}={strat_value}")

    map_title = " • ".join(title_bits)

# --- base layer: states (context) ---
state_base = px.choropleth(
    states_attr,
    geojson=states_geojson,
    locations="statefp",
    featureidkey="properties.statefp",
    color_discrete_sequence=["#CFCFCF"],
)
state_base.update_traces(marker_line_width=0.6, marker_opacity=0.25, hoverinfo="skip")

# --- top layer: counties (data) ---
fig = px.choropleth(
    agg_map,
    geojson=counties_geojson,
    locations="fips",
    featureidkey="properties.fips",
    color="value",
    color_continuous_scale="YlOrRd",
    hover_name="namelsad" if "namelsad" in agg_map.columns else ("name" if "name" in agg_map.columns else None),
    hover_data={
        "fips": True,
        "deaths": True,
        "population": True,
        "rate_per_100k": (metric_key == "rate"),
        "value": False,
    },
    title=map_title,
)

# Combine layers
for tr in fig.data:
    state_base.add_trace(tr)

state_base.update_geos(fitbounds="locations", visible=False)
state_base.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=650)
st.plotly_chart(state_base, use_container_width=True)

with right:
    st.subheader("Top counties")

    top = agg_map.dropna(subset=["value"]).sort_values("value", ascending=False).head(top_n).copy()
    show_cols = ["fips"]
    for c in ["namelsad", "name"]:
        if c in top.columns and c not in show_cols:
            show_cols.insert(0, c)
            break
    show_cols += ["deaths", "population", "rate_per_100k"]
    top = top[show_cols]
    st.dataframe(top, use_container_width=True, height=420)

    st.subheader("Quick summaries")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Counties shown", f"{agg_map['fips'].nunique():,}")
    with c2:
        st.metric("Total deaths (filtered)", f"{int(np.nansum(agg_map['deaths'])):,}")
    

# -----------------------------
# Extra charts
# -----------------------------
st.divider()
st.subheader("Exploration")

cA, cB = st.columns([1.2, 1.0], gap="large")

with cA:
    if "year" in df.columns and df["year"].notna().any():
        trend = (
            df.groupby("year", as_index=False)
              .agg(deaths=("deaths", "sum"), population=("population", "sum"))
        )
        trend["rate_per_100k"] = (trend["deaths"] / trend["population"]) * 100_000.0
        ycol = "rate_per_100k" if metric_key == "rate" else "deaths"
        tfig = px.line(trend, x="year", y=ycol, markers=True,
                       title=("Trend: rate per 100k" if metric_key == "rate" else "Trend: death burden"))
        tfig.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(tfig, use_container_width=True)
    else:
        st.info("No 'year' column available for this selection, so the trend chart is hidden.")

with cB:
    if "mcd_type" in df.columns and (mcd_sel is None or mcd_sel == "All"):
        breakdown = (
            df.groupby("mcd_type", as_index=False)
              .agg(deaths=("deaths", "sum"), population=("population", "sum"))
        )
        breakdown["rate_per_100k"] = (breakdown["deaths"] / breakdown["population"]) * 100_000.0
        ycol = "rate_per_100k" if metric_key == "rate" else "deaths"
        bfig = px.bar(breakdown.sort_values(ycol, ascending=False), x="mcd_type", y=ycol,
                      title=("Substance breakdown: rate per 100k" if metric_key == "rate" else "Substance breakdown: deaths"))
        bfig.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(bfig, use_container_width=True)
    else:
        st.info("Select 'All' substances to see a substance breakdown chart.")

st.caption(
    "Default view: Overall dataset • All substances • Rate per 100k. "
    "Map excludes Alaska, Hawaii, and U.S. territories (AS, GU, MP, PR, VI) to keep the contiguous U.S. view stable. "
    "Rates are computed from summed deaths and summed population across the selected filters."
)