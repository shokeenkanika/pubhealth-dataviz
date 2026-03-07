# ==============================
# spatial_policyquestionfigures.py
# ==============================

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
dataverse_raw = data_path / "raw" / "dataverse_files"
derived_dir = data_path / "derived"
geo_derived = data_path / "derived" / "geo"
fig_dir = project_root / "outputs" / "figures" / "policy_maps"

geo_derived.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)
derived_dir.mkdir(parents=True, exist_ok=True)

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
# 1) Load TIGER counties + exclude AK/HI/territories
# -----------------------------------------------------------------------------
tiger_counties_path = tiger_raw / "tl_2025_us_county.shp"
counties = gpd.read_file(tiger_counties_path)
counties.columns = [_snake_case(c) for c in counties.columns]

geoid_col = "geoid" if "geoid" in counties.columns else ("geoid10" if "geoid10" in counties.columns else None)
if geoid_col is None:
    raise KeyError("Could not find GEOID column (expected 'GEOID' or 'GEOID10') in TIGER county shapefile.")

counties["fips"] = _standardize_fips(counties[geoid_col])
counties = counties.to_crs(epsg=4326)

exclude_statefps = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, AS, GU, MP, PR, VI
counties = counties.loc[~counties["fips"].str[:2].isin(exclude_statefps)].copy()

counties_aea = counties.to_crs(epsg=5070)
if "statefp" not in counties_aea.columns:
    counties_aea["statefp"] = counties_aea["fips"].str[:2]
states_aea = counties_aea.dissolve(by="statefp")
states = states_aea.to_crs(epsg=4326)

# -----------------------------------------------------------------------------
# 2) Load CDC overall and aggregate across time (ignore year)
# -----------------------------------------------------------------------------
overall_csv = cdc_derived / "overall_county_2018_2023_all_mcd_types.csv"
cdc = pd.read_csv(overall_csv, dtype=str)
cdc.columns = [_snake_case(c) for c in cdc.columns]
cdc["fips"] = _standardize_fips(cdc["fips"])
cdc["statefp"] = cdc["fips"].str[:2]

for c in ["deaths", "population", "year"]:
    if c in cdc.columns:
        cdc[c] = pd.to_numeric(cdc[c], errors="coerce")

# County x substance totals
agg_by_mcd = (
    cdc.groupby(["fips", "statefp", "mcd_type"], as_index=False)
       .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
agg_by_mcd["rate_per_100k"] = (agg_by_mcd["deaths"] / agg_by_mcd["population"]) * 100_000.0
agg_by_mcd.loc[agg_by_mcd["population"].isna() | (agg_by_mcd["population"] == 0), "rate_per_100k"] = np.nan

# County totals (all substances combined)
agg_all = (
    cdc.groupby(["fips", "statefp"], as_index=False)
       .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
agg_all["rate_per_100k"] = (agg_all["deaths"] / agg_all["population"]) * 100_000.0
agg_all.loc[agg_all["population"].isna() | (agg_all["population"] == 0), "rate_per_100k"] = np.nan

# -----------------------------------------------------------------------------
# 3) Load policy dataset (Excel) OR load cached minimal derived policy file
#    Use sheet: "Full Data"
# -----------------------------------------------------------------------------
policy_min_path = derived_dir / "policy_state_2018_2023_minimal.csv"

if policy_min_path.exists():
    policy_state = pd.read_csv(policy_min_path, dtype=str)
    policy_state.columns = [_snake_case(c) for c in policy_state.columns]
    policy_state["statefp"] = policy_state["statefp"].astype(str).str.zfill(2)
    print(policy_state["policy_group"].value_counts(dropna=False))

    # ensure numeric where needed
    for c in ["mml_approved", "rec_cann_approved", "policy_group"]:
        if c in policy_state.columns:
            policy_state[c] = pd.to_numeric(policy_state[c], errors="coerce").fillna(0).astype(int)

else:
    policy_xlsx = dataverse_raw / "Cannabis Policies Full Dataset 81425.xlsx"
    policy = pd.read_excel(policy_xlsx, sheet_name="Full Data")
    policy.columns = [_snake_case(c) for c in policy.columns]

    if "fips" not in policy.columns or "year" not in policy.columns:
        raise KeyError("Policy sheet must include 'FIPS' and 'Year' columns (check sheet name and headers).")
    
    policy["statefp"] = pd.to_numeric(policy["fips"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    policy["year"] = pd.to_numeric(policy["year"], errors="coerce").astype("Int64")

    for col in ["mml_approved", "rec_cann_approved"]:
        if col in policy.columns:
            policy[col] = pd.to_numeric(policy[col], errors="coerce").fillna(0).astype(int)
        else:
            raise KeyError(f"Missing required policy column: {col}")

    # Restrict to 2018–2023 (matches your CDC window)
    policy_18_23 = policy.loc[policy["year"].between(2018, 2023)].copy()

    # Collapse to a single status per state for this period (any-year approval -> 1)
    policy_state = (
        policy_18_23.groupby("statefp", as_index=False)
                   .agg(mml_approved=("mml_approved", "max"),
                        rec_cann_approved=("rec_cann_approved", "max"))
    )

    # Create a simple 3-level policy group:
    # 0 = neither, 1 = medical only, 2 = adult-use (any)
    policy_state["policy_group"] = np.select(
        [
            (policy_state["rec_cann_approved"] == 1),
            (policy_state["rec_cann_approved"] == 0) & (policy_state["mml_approved"] == 1),
        ],
        [2, 1],
        default=0,
    ).astype(int)

    policy_state["policy_group_label"] = policy_state["policy_group"].map({
        0: "No medical/adult-use",
        1: "Medical only",
        2: "Adult-use (any)",
    })

    # Save minimal policy file for reuse (in derived)
    policy_state.to_csv(policy_min_path, index=False)

# -----------------------------------------------------------------------------
# 4) Join policy -> county outcomes -> geometry (PATCHED: fill missing groups)
# -----------------------------------------------------------------------------
policy_state["statefp"] = policy_state["statefp"].astype(str).str.zfill(2)

agg_by_mcd = agg_by_mcd.merge(policy_state, on="statefp", how="left")
agg_all = agg_all.merge(policy_state, on="statefp", how="left")

# PATCH: avoid NaN policy_group breaking subsets (and later plotting)
label_map = {0: "No medical/adult-use", 1: "Medical only", 2: "Adult-use (any)"}

agg_by_mcd["policy_group"] = agg_by_mcd["policy_group"].fillna(0).astype(int)
agg_all["policy_group"] = agg_all["policy_group"].fillna(0).astype(int)

agg_by_mcd["policy_group_label"] = agg_by_mcd["policy_group"].map(label_map)
agg_all["policy_group_label"] = agg_all["policy_group"].map(label_map)

map_by_mcd = counties.merge(agg_by_mcd, on="fips", how="left")
map_all = counties.merge(agg_all, on="fips", how="left")

# -----------------------------------------------------------------------------
# 5) Policy analysis figures (simple: maps stratified by policy group)
#    (PATCHED: robust plot to prevent aspect errors)
# -----------------------------------------------------------------------------
def _policy_choropleth(gdf: gpd.GeoDataFrame, value_col: str, title: str, out_png: Path) -> None:
    # Drop missing/invalid geometry (prevents NaN bounds -> aspect error)
    gdf = gdf.loc[gdf.geometry.notna()].copy()
    if len(gdf) == 0:
        print(f"SKIP (no geometry): {title}")
        return

    try:
        gdf = gdf.loc[gdf.is_valid].copy()
    except Exception:
        pass

    if len(gdf) == 0:
        print(f"SKIP (no valid geometry): {title}")
        return

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
    ax.set_title(title, fontsize=15, pad=12)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

mcd_types = sorted([x for x in map_by_mcd["mcd_type"].dropna().unique().tolist()])

# All substances: rate & burden maps by policy group
for pg, pg_label in [(0, "No medical/adult-use"), (1, "Medical only"), (2, "Adult-use (any)")]:
    g = map_all.loc[map_all["policy_group"] == pg].copy()

    _policy_choropleth(
        g,
        value_col="rate_per_100k",
        title=f"Policy layer: overdose mortality rate per 100k (2018–2023 aggregated) | {pg_label}",
        out_png=fig_dir / f"POLICY_allsubstances_rate_pg{pg}.png",
    )
    _policy_choropleth(
        g,
        value_col="deaths",
        title=f"Policy layer: overdose mortality burden (total deaths, 2018–2023 aggregated) | {pg_label}",
        out_png=fig_dir / f"POLICY_allsubstances_burden_pg{pg}.png",
    )

# By substance: rate & burden maps by policy group
for m in mcd_types:
    sub = map_by_mcd.loc[map_by_mcd["mcd_type"] == m].copy()
    for pg, pg_label in [(0, "No medical/adult-use"), (1, "Medical only"), (2, "Adult-use (any)")]:
        g = sub.loc[sub["policy_group"] == pg].copy()

        _policy_choropleth(
            g,
            value_col="rate_per_100k",
            title=f"Policy layer: overdose mortality rate per 100k (2018–2023 aggregated) | {m} | {pg_label}",
            out_png=fig_dir / f"POLICY_{m}_rate_pg{pg}.png",
        )
        _policy_choropleth(
            g,
            value_col="deaths",
            title=f"Policy layer: overdose mortality burden (total deaths, 2018–2023 aggregated) | {m} | {pg_label}",
            out_png=fig_dir / f"POLICY_{m}_burden_pg{pg}.png",
        )

print("Saved policy figures to:", fig_dir)
print("Saved/loaded minimal policy file:", policy_min_path)