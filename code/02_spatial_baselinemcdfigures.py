# =========================
# spatial_baselinemcdfigures.py
# =========================

import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# Please change paths as per your folder organization before re-running this file
project_root = Path('/Users/koniks/Desktop/GitHub Folder/pubhealth-dataviz')
data_path = project_root / "data"
cdc_derived = data_path / "derived"
tiger_raw = data_path / "raw" / "spatial-county"
geo_derived = data_path / "derived" / "geo"
fig_dir = project_root / "outputs" / "figures" / "baseline_death_maps"
geo_derived.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

def _snake_case(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("%", "pct").replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _standardize_fips(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

# -----------------------------------------------------------------------------
# 1) Load TIGER county shapefile
# -----------------------------------------------------------------------------
tiger_counties_path = tiger_raw / "tl_2025_us_county.shp" 
counties = gpd.read_file(tiger_counties_path)
counties.columns = [_snake_case(c) for c in counties.columns]

geoid_col = "geoid" if "geoid" in counties.columns else ("geoid10" if "geoid10" in counties.columns else None)
if geoid_col is None:
    raise KeyError("Could not find GEOID column (expected 'GEOID' or 'GEOID10') in TIGER county shapefile.")

counties["fips"] = _standardize_fips(counties[geoid_col])
counties = counties.to_crs(epsg=4326)

# --- add right after you create counties["fips"] ---
# Keep Lower 48 + DC, exclude AK, HI, and territories
exclude_statefps = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, AS, GU, MP, PR, VI
counties = counties.loc[~counties["fips"].str[:2].isin(exclude_statefps)].copy()

# Recompute these after filtering so state outlines/insets don't include excluded areas
counties_aea = counties.to_crs(epsg=5070)
if "statefp" not in counties_aea.columns:
    counties_aea["statefp"] = counties_aea["fips"].str[:2]
states_aea = counties_aea.dissolve(by="statefp")
states = states_aea.to_crs(epsg=4326)

# -----------------------------------------------------------------------------
# 2) Load CDC derived overall file and aggregate across time (ignore year)
# -----------------------------------------------------------------------------
overall_csv = cdc_derived / "overall_county_2018_2023_all_mcd_types.csv"
cdc = pd.read_csv(overall_csv, dtype=str)
cdc.columns = [_snake_case(c) for c in cdc.columns]
cdc["fips"] = _standardize_fips(cdc["fips"])

for c in ["deaths", "population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper", "year"]:
    if c in cdc.columns:
        cdc[c] = pd.to_numeric(cdc[c], errors="coerce")

# Compute county x substance totals across years
agg_by_mcd = (
    cdc.groupby(["fips", "mcd_type"], as_index=False)
       .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
agg_by_mcd["rate_per_100k"] = (agg_by_mcd["deaths"] / agg_by_mcd["population"]) * 100_000.0
agg_by_mcd.loc[agg_by_mcd["population"].isna() | (agg_by_mcd["population"] == 0), "rate_per_100k"] = np.nan

# All substances combined
agg_all = (
    cdc.groupby(["fips"], as_index=False)
       .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
agg_all["rate_per_100k"] = (agg_all["deaths"] / agg_all["population"]) * 100_000.0
agg_all.loc[agg_all["population"].isna() | (agg_all["population"] == 0), "rate_per_100k"] = np.nan

# Join to geometry
map_by_mcd = counties.merge(agg_by_mcd, on="fips", how="left")
map_all = counties.merge(agg_all, on="fips", how="left")

# -----------------------------------------------------------------------------
# 3) Base choropleths (no labels)
#    Make BOTH: rate maps and burden maps
# -----------------------------------------------------------------------------
def _base_choropleth(gdf: gpd.GeoDataFrame, value_col: str, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_axis_off()

    gdf.plot(
        column=value_col,
        ax=ax,
        legend=True,
        scheme="Quantiles",
        k=7,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        linewidth=0.05,
    )

    states.boundary.plot(ax=ax, linewidth=0.6)
    ax.set_title(title, fontsize=16, pad=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

mcd_types = sorted([x for x in map_by_mcd["mcd_type"].dropna().unique().tolist()])

# Base maps: by substance (rate and burden)
for m in mcd_types:
    sub = map_by_mcd.loc[map_by_mcd["mcd_type"] == m].copy()

    _base_choropleth(
        sub,
        value_col="rate_per_100k",
        title=f"County-level overdose mortality rate per 100k (2018–2023 aggregated): {m}",
        out_png=fig_dir / f"BASE_rate_per_100k_{m}.png",
    )
    _base_choropleth(
        sub,
        value_col="deaths",
        title=f"County-level overdose mortality burden (total deaths, 2018–2023 aggregated): {m}",
        out_png=fig_dir / f"BASE_death_burden_{m}.png",
    )

# Base maps: all substances (rate and burden)
_base_choropleth(
    map_all,
    value_col="rate_per_100k",
    title="County-level overdose mortality rate per 100k (2018–2023 aggregated): All listed substances",
    out_png=fig_dir / "BASE_rate_per_100k_all_substances.png",
)
_base_choropleth(
    map_all,
    value_col="deaths",
    title="County-level overdose mortality burden (total deaths, 2018–2023 aggregated): All listed substances",
    out_png=fig_dir / "BASE_death_burden_all_substances.png",
)

# -----------------------------------------------------------------------------
# 4) Analog style maps + label top 5 counties by incidence (separate from base maps)
#    (Make BOTH: rates and burden maps, with requested titles)
# -----------------------------------------------------------------------------
def _choropleth(gdf: gpd.GeoDataFrame, value_col: str, title: str, out_png: Path) -> None:
    # Work on a copy + ensure geometry for labels
    plot_gdf = gdf.copy()

    # Identify top 5 counties (by incidence). If you truly mean raw deaths, switch to "deaths".
    top5 = plot_gdf.dropna(subset=[value_col]).nlargest(5, value_col).copy()

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_axis_off()

    # "Analog" look: soft background + subtle county lines + stronger state boundaries
    ax.set_facecolor("#f4f1ea")  # paper-like background

    plot_gdf.plot(
        column=value_col,
        ax=ax,
        legend=True,
        scheme="Quantiles",
        k=7,
        linewidth=0.08,
        edgecolor="white",
        missing_kwds={"color": "lightgrey", "label": "Missing"},
    )

    # State boundaries (clean and readable)
    states.boundary.plot(ax=ax, linewidth=0.8, color="black", alpha=0.55)

    # Label top 5 at county centroid (in map CRS)
    # Use representative_point() so labels fall inside polygons
    top5_pts = top5.copy()
    top5_pts["label_pt"] = top5_pts.geometry.representative_point()
    for _, r in top5_pts.iterrows():
        x, y = r["label_pt"].x, r["label_pt"].y
        name = r["county"] if "county" in top5_pts.columns else r.get("name", "")
        val = r[value_col]
        ax.annotate(
            f"{name}\n{val:.1f}",
            xy=(x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    ax.set_title(title, fontsize=16, pad=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Analog maps: by substance (rate and burden) with requested titles
for m in mcd_types:
    sub = map_by_mcd.loc[map_by_mcd["mcd_type"] == m].copy()

    _choropleth(
        sub,
        value_col="rate_per_100k",
        title=f"baseline geography: where are overdose death rates highest for {m}?",
        out_png=fig_dir / f"ANALOG_top5_rate_{m}.png",
    )
    _choropleth(
        sub,
        value_col="deaths",
        title=f"baseline geography: where are overdose death burden highest for {m}?",
        out_png=fig_dir / f"ANALOG_top5_burden_{m}.png",
    )

# Analog maps: all substances combined (rate and burden) with requested titles
_choropleth(
    map_all,
    value_col="rate_per_100k",
    title="baseline geography: where are overdose death rates highest for all listed substances?",
    out_png=fig_dir / "ANALOG_top5_rate_all_substances.png",
)
_choropleth(
    map_all,
    value_col="deaths",
    title="baseline geography: where are overdose death burden highest for all listed substances?",
    out_png=fig_dir / "ANALOG_top5_burden_all_substances.png",
)
# -----------------------------------------------------------------------------
# 5) (Optional) Save GeoPackages for dashboard use (baseline_mcd figures)
# -----------------------------------------------------------------------------
out_gpkg_by_mcd = geo_derived / "cdc_overall_county_by_mcd_joined.gpkg"
out_gpkg_all = geo_derived / "cdc_overall_county_allsubstances_joined.gpkg"

map_by_mcd.to_file(out_gpkg_by_mcd, layer="counties_by_mcd_type", driver="GPKG")
map_all.to_file(out_gpkg_all, layer="counties_all_substances", driver="GPKG")

# Quick sanity prints (kept minimal)
print("Saved maps to:", fig_dir)
print("Saved GeoPackages to:", out_gpkg_by_mcd, "and", out_gpkg_all)