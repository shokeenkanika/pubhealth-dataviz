# Purpose:
# Static (Altair) exploratory figures to support the policy question:
# State cannabis policy environment (3-category) × county overdose mortality outcomes, plus heterogeneity (sex + urbanization)

import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

# Altair defaults can choke on large datasets unless disabled
alt.data_transformers.disable_max_rows()

# Please change paths as per your folder organization before re-running this file
project_root = Path('/Users/koniks/Desktop/GitHub Folder/pubhealth-dataviz')
data_path = project_root / "data"
cdc_derived = data_path / "derived"
dataverse_raw = data_path / "raw" / "dataverse_files"
out_dir = project_root / "outputs" / "figures" / "altair_figures"
out_dir.mkdir(parents=True, exist_ok=True)

# CDC inputs (from preprocessing outputs)
overall_csv = cdc_derived / "overall_county_2018_2023_all_mcd_types.csv"
sex_csv = cdc_derived / "county_sex_2018_2023_all_mcd_types.csv"
urban_csv = cdc_derived / "county_urbanization2023_all_mcd_types.csv"

# Policy inputs
policy_min_csv = cdc_derived / "policy_state_2018_2023_minimal.csv"  # created by spatial_policyquestionfigures.py
policy_xlsx = dataverse_raw / "Cannabis Policies Full Dataset 81425.xlsx"  # fallback if minimal not found

# -----------------------------
# Helpers
# -----------------------------
def _snake_case(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("%", "pct").replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _std_county_fips(x: pd.Series) -> pd.Series:
    return (
        x.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

def _std_statefp_from_policy_fips(series: pd.Series) -> pd.Series:
    # Policy dataset "FIPS" is state-level (e.g., 1 for AL). Convert to 2-digit.
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str).str.zfill(2)

def _compute_rate(df: pd.DataFrame, deaths_col="deaths", pop_col="population") -> pd.Series:
    rate = (df[deaths_col] / df[pop_col]) * 100_000.0
    rate = rate.where((df[pop_col].notna()) & (df[pop_col] > 0), np.nan)
    return rate

def _safe_save(chart: alt.Chart, out_path_base: Path) -> None:
    """
    Saves HTML always. Tries PNG if altair_saver installed.
    """
    html_path = out_path_base.with_suffix(".html")
    chart.save(str(html_path))

    try:
        # Optional dependency
        from altair_saver import save as alt_save  # type: ignore
        png_path = out_path_base.with_suffix(".png")
        alt_save(chart, str(png_path))
    except Exception:
        pass

# -----------------------------
# Load policy state table
# -----------------------------
if policy_min_csv.exists():
    policy_state = pd.read_csv(policy_min_csv, dtype=str)
    policy_state.columns = [_snake_case(c) for c in policy_state.columns]
    policy_state["statefp"] = policy_state["statefp"].astype(str).str.zfill(2)
    for c in ["mml_approved", "rec_cann_approved", "policy_group"]:
        if c in policy_state.columns:
            policy_state[c] = pd.to_numeric(policy_state[c], errors="coerce").fillna(0).astype(int)
    if "policy_group_label" not in policy_state.columns:
        label_map = {0: "No medical/adult-use", 1: "Medical only", 2: "Adult-use (any)"}
        policy_state["policy_group_label"] = policy_state["policy_group"].map(label_map)
else:
    # Build minimal policy from the Excel (Full Data) and cache it to derived/
    pol = pd.read_excel(policy_xlsx, sheet_name="Full Data")
    pol.columns = [_snake_case(c) for c in pol.columns]
    pol["statefp"] = _std_statefp_from_policy_fips(pol["fips"])
    pol["year"] = pd.to_numeric(pol["year"], errors="coerce")

    for c in ["mml_approved", "rec_cann_approved"]:
        if c not in pol.columns:
            raise KeyError(f"Missing required policy column: {c}")
        pol[c] = pd.to_numeric(pol[c], errors="coerce").fillna(0).astype(int)

    pol = pol.loc[pol["year"].between(2018, 2023)].copy()

    policy_state = (
        pol.groupby("statefp", as_index=False)
           .agg(mml_approved=("mml_approved", "max"), rec_cann_approved=("rec_cann_approved", "max"))
    )

    policy_state["policy_group"] = np.select(
        [
            (policy_state["rec_cann_approved"] == 1),
            (policy_state["rec_cann_approved"] == 0) & (policy_state["mml_approved"] == 1),
        ],
        [2, 1],
        default=0,
    ).astype(int)

    label_map = {0: "No medical/adult-use", 1: "Medical only", 2: "Adult-use (any)"}
    policy_state["policy_group_label"] = policy_state["policy_group"].map(label_map)

    policy_state.to_csv(policy_min_csv, index=False)


# -----------------------------
# Load CDC overall (county x year x mcd_type)
# -----------------------------
overall = pd.read_csv(overall_csv, dtype=str)
overall.columns = [_snake_case(c) for c in overall.columns]
overall["fips"] = _std_county_fips(overall["fips"])
overall["statefp"] = overall["fips"].str[:2]

for c in ["year", "deaths", "population"]:
    overall[c] = pd.to_numeric(overall[c], errors="coerce")

# Join policy group to county rows
overall = overall.merge(policy_state[["statefp", "policy_group", "policy_group_label"]], on="statefp", how="left")
overall["policy_group"] = overall["policy_group"].fillna(0).astype(int)
overall["policy_group_label"] = overall["policy_group_label"].fillna("No medical/adult-use")


# =============================================================================
# TABLES: identify high/low burden counties + high/low burden states
# =============================================================================
county_totals = (
    overall.groupby(["fips", "county", "statefp"], as_index=False)
           .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
county_totals["rate_per_100k"] = _compute_rate(county_totals)

state_totals = (
    overall.groupby(["statefp"], as_index=False)
           .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
state_totals["rate_per_100k"] = _compute_rate(state_totals)
state_totals = state_totals.merge(policy_state[["statefp", "policy_group_label"]], on="statefp", how="left")

top5_counties = county_totals.sort_values("deaths", ascending=False).head(5)
bot5_counties = county_totals.sort_values("deaths", ascending=True).head(5)

top5_states = state_totals.sort_values("deaths", ascending=False).head(5)
bot5_states = state_totals.sort_values("deaths", ascending=True).head(5)

print("\nTop 5 counties by death burden (2018–2023, all substances & years summed):")
print(top5_counties[["fips", "county", "deaths", "population", "rate_per_100k"]].to_string(index=False))

print("\nBottom 5 counties by death burden (2018–2023, all substances & years summed):")
print(bot5_counties[["fips", "county", "deaths", "population", "rate_per_100k"]].to_string(index=False))

print("\nTop 5 states by death burden (2018–2023, all substances & years summed):")
print(top5_states[["statefp", "policy_group_label", "deaths", "population", "rate_per_100k"]].to_string(index=False))

print("\nBottom 5 states by death burden (2018–2023, all substances & years summed):")
print(bot5_states[["statefp", "policy_group_label", "deaths", "population", "rate_per_100k"]].to_string(index=False))


# =============================================================================
# FIGURE 1: Bar plot — mean county rate per 100k by policy group and substance
# =============================================================================
by_policy_substance = (
    overall.groupby(["policy_group_label", "mcd_type", "fips"], as_index=False)
           .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
by_policy_substance["rate_per_100k"] = _compute_rate(by_policy_substance)

# Collapse to mean county rate within each policy group x substance
fig1_df = (
    by_policy_substance.groupby(["policy_group_label", "mcd_type"], as_index=False)
                       .agg(mean_county_rate=("rate_per_100k", "mean"))
)

fig1 = (
    alt.Chart(fig1_df)
    .mark_bar()
    .encode(
        x=alt.X("mcd_type:N", title="Multiple cause of death type"),
        y=alt.Y("mean_county_rate:Q", title="Mean county rate per 100k (2018–2023 summed then averaged across counties)"),
        column=alt.Column("policy_group_label:N", title="Policy category"),
        tooltip=["policy_group_label", "mcd_type", alt.Tooltip("mean_county_rate:Q", format=".2f")],
    )
    .properties(title="Policy category × substance: mean county overdose mortality rate per 100k", width=210, height=240)
)

_safe_save(fig1, out_dir / "fig1_policy_by_substance_mean_county_rate")


# =============================================================================
# FIGURE 2: Histogram — distribution of county rates within each policy group (choose one substance)
# =============================================================================
SUBSTANCE_FOR_HIST = "T404_synthopioids"  # change if you want
hist_df = by_policy_substance.loc[by_policy_substance["mcd_type"] == SUBSTANCE_FOR_HIST].copy()
hist_df = hist_df.loc[hist_df["rate_per_100k"].notna()].copy()

# winsorize for nicer histograms (optional)
q99 = hist_df["rate_per_100k"].quantile(0.99)
hist_df["rate_per_100k_clip"] = hist_df["rate_per_100k"].clip(upper=q99)

fig2 = (
    alt.Chart(hist_df)
    .mark_bar(opacity=0.85)
    .encode(
        x=alt.X("rate_per_100k_clip:Q", bin=alt.Bin(maxbins=35), title=f"{SUBSTANCE_FOR_HIST} rate per 100k (clipped at 99th pct)"),
        y=alt.Y("count():Q", title="Number of counties"),
        color=alt.Color("policy_group_label:N", title="Policy category"),
        tooltip=[alt.Tooltip("count():Q", title="Count")],
    )
    .properties(title=f"Distribution of county rates: {SUBSTANCE_FOR_HIST} by policy category", width=800, height=320)
)

_safe_save(fig2, out_dir / "fig2_hist_county_rates_by_policy")


# =============================================================================
# FIGURE 3: Scatter — state-level relationship (policy group vs death burden & rate)
# =============================================================================
# State-level totals by substance
state_sub = (
    overall.groupby(["statefp", "policy_group_label", "mcd_type"], as_index=False)
           .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
state_sub["rate_per_100k"] = _compute_rate(state_sub)

# Example scatter: burden vs rate, colored by policy, faceted by substance
fig3 = (
    alt.Chart(state_sub)
    .mark_circle(size=80, opacity=0.75)
    .encode(
        x=alt.X("rate_per_100k:Q", title="State rate per 100k (2018–2023 aggregated)"),
        y=alt.Y("deaths:Q", title="State death burden (total deaths, 2018–2023)"),
        color=alt.Color("policy_group_label:N", title="Policy category"),
        tooltip=["statefp", "policy_group_label", "mcd_type", "deaths", alt.Tooltip("rate_per_100k:Q", format=".2f")],
        facet=alt.Facet("mcd_type:N", columns=3, title="Substance"),
    )
    .properties(title="State outcomes: rate vs burden, by substance and policy category", width=260, height=240)
)

_safe_save(fig3, out_dir / "fig3_state_scatter_rate_vs_burden")


# =============================================================================
# FIGURE 4: Sex heterogeneity — Male vs Female rate by policy group (all substances combined)
# =============================================================================
sex_df = pd.read_csv(sex_csv, dtype=str)
sex_df.columns = [_snake_case(c) for c in sex_df.columns]
sex_df["fips"] = _std_county_fips(sex_df["fips"])
sex_df["statefp"] = sex_df["fips"].str[:2]
for c in ["year", "deaths", "population"]:
    sex_df[c] = pd.to_numeric(sex_df[c], errors="coerce")

sex_df = sex_df.merge(policy_state[["statefp", "policy_group_label"]], on="statefp", how="left")
sex_df["policy_group_label"] = sex_df["policy_group_label"].fillna("No medical/adult-use")

sex_agg = (
    sex_df.groupby(["policy_group_label", "sex"], as_index=False)
          .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
sex_agg["rate_per_100k"] = _compute_rate(sex_agg)

fig4 = (
    alt.Chart(sex_agg)
    .mark_bar()
    .encode(
        x=alt.X("policy_group_label:N", title="Policy category"),
        y=alt.Y("rate_per_100k:Q", title="Rate per 100k (all substances, 2018–2023 aggregated)"),
        color=alt.Color("sex:N", title="Sex"),
        column=alt.Column("sex:N", title=None),
        tooltip=["policy_group_label", "sex", alt.Tooltip("rate_per_100k:Q", format=".2f")],
    )
    .properties(title="Sex heterogeneity: rates by policy category (all substances combined)", width=250, height=280)
)

_safe_save(fig4, out_dir / "fig4_sex_by_policy_rates")


# =============================================================================
# FIGURE 5: Urbanization heterogeneity — mean county rate by policy group × urbanization
# =============================================================================
urb = pd.read_csv(urban_csv, dtype=str)
urb.columns = [_snake_case(c) for c in urb.columns]
urb["fips"] = _std_county_fips(urb["fips"])
urb["statefp"] = urb["fips"].str[:2]
for c in ["year", "deaths", "population"]:
    urb[c] = pd.to_numeric(urb[c], errors="coerce")

# Column name in your file after snake_case is typically "2023_urbanization"
urb_col = "2023_urbanization" if "2023_urbanization" in urb.columns else "urbanization"

urb = urb.merge(policy_state[["statefp", "policy_group_label"]], on="statefp", how="left")
urb["policy_group_label"] = urb["policy_group_label"].fillna("No medical/adult-use")

urb_agg = (
    urb.groupby(["policy_group_label", urb_col, "fips"], as_index=False)
       .agg(deaths=("deaths", "sum"), population=("population", "sum"))
)
urb_agg["rate_per_100k"] = _compute_rate(urb_agg)

heat_df = (
    urb_agg.groupby(["policy_group_label", urb_col], as_index=False)
           .agg(mean_county_rate=("rate_per_100k", "mean"))
)

fig5 = (
    alt.Chart(heat_df)
    .mark_rect()
    .encode(
        x=alt.X(f"{urb_col}:N", title="Urbanization (2023)"),
        y=alt.Y("policy_group_label:N", title="Policy category"),
        color=alt.Color("mean_county_rate:Q", title="Mean county rate per 100k", scale=alt.Scale(scheme="yelloworangered")),
        tooltip=["policy_group_label", urb_col, alt.Tooltip("mean_county_rate:Q", format=".2f")],
    )
    .properties(title="Urbanization × policy category: mean county overdose mortality rate per 100k", width=800, height=180)
)

_safe_save(fig5, out_dir / "fig5_urbanization_policy_heatmap")

# =============================================================================
# FIGURE 6 (revised): 36 separate files
# 6 drug types × 3 policy groups × 2 (highest burden / lowest burden county)
#
# For each (drug × policy_group) combination:
#   - Rank counties by TOTAL DEATHS (2018–2023 summed) for that specific drug
#   - Select the #1 highest and #1 lowest (with >0 deaths to avoid empty panels)
#   - Plot that county's year-by-year rate per 100k trend
#   - Overlay a red vertical rule at the first adult-use year (if applicable)
#
# Output naming: fig6_{mcd_type}_{policy_slug}_{high|low}_burden.html
# =============================================================================

import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

alt.data_transformers.disable_max_rows()

# ── paths (adjust to match your environment) ─────────────────────────────────
data_path = Path('/Users/koniks/Desktop/GitHub Folder/pubhealth-dataviz/data')
cdc_derived    = data_path / "derived"
dataverse_raw  = data_path / "raw" / "dataverse_files"
out_dir        = data_path / "outputs" / "figures" / "altair_figures"
out_dir.mkdir(parents=True, exist_ok=True)

overall_csv    = cdc_derived / "overall_county_2018_2023_all_mcd_types.csv"
policy_min_csv = cdc_derived / "policy_state_2018_2023_minimal.csv"
policy_xlsx    = dataverse_raw / "Cannabis Policies Full Dataset 81425.xlsx"

# ── helpers ───────────────────────────────────────────────────────────────────
def _snake_case(s):
    s = str(s).strip().lower()
    for old, new in [("%","pct"),("/","_"),("-","_"),(" ","_")]:
        s = s.replace(old, new)
    while "__" in s:
        s = s.replace("__","_")
    return s

def _std_county_fips(x):
    return (x.astype(str).str.strip()
             .str.replace(r"\.0$","",regex=True)
             .str.zfill(5))

def _std_statefp_from_policy_fips(series):
    return (pd.to_numeric(series, errors="coerce")
              .astype("Int64").astype(str).str.zfill(2))

def _compute_rate(df, deaths_col="deaths", pop_col="population"):
    rate = (df[deaths_col] / df[pop_col]) * 100_000.0
    return rate.where((df[pop_col].notna()) & (df[pop_col] > 0), np.nan)

def _safe_save(chart, out_path_base):
    html_path = out_path_base.with_suffix(".html")
    chart.save(str(html_path))
    try:
        from altair_saver import save as alt_save
        alt_save(chart, str(out_path_base.with_suffix(".png")))
    except Exception:
        pass

# ── load policy state table ───────────────────────────────────────────────────
if policy_min_csv.exists():
    policy_state = pd.read_csv(policy_min_csv, dtype=str)
    policy_state.columns = [_snake_case(c) for c in policy_state.columns]
    policy_state["statefp"] = policy_state["statefp"].astype(str).str.zfill(2)
    for c in ["mml_approved","rec_cann_approved","policy_group"]:
        if c in policy_state.columns:
            policy_state[c] = pd.to_numeric(policy_state[c], errors="coerce").fillna(0).astype(int)
    if "policy_group_label" not in policy_state.columns:
        label_map = {0:"No medical/adult-use", 1:"Medical only", 2:"Adult-use (any)"}
        policy_state["policy_group_label"] = policy_state["policy_group"].map(label_map)
else:
    pol = pd.read_excel(policy_xlsx, sheet_name="Full Data")
    pol.columns = [_snake_case(c) for c in pol.columns]
    pol["statefp"] = _std_statefp_from_policy_fips(pol["fips"])
    pol["year"] = pd.to_numeric(pol["year"], errors="coerce")
    for c in ["mml_approved","rec_cann_approved"]:
        pol[c] = pd.to_numeric(pol[c], errors="coerce").fillna(0).astype(int)
    pol = pol.loc[pol["year"].between(2018,2023)].copy()
    policy_state = (pol.groupby("statefp", as_index=False)
                       .agg(mml_approved=("mml_approved","max"),
                            rec_cann_approved=("rec_cann_approved","max")))
    policy_state["policy_group"] = np.select(
        [(policy_state["rec_cann_approved"]==1),
         (policy_state["rec_cann_approved"]==0) & (policy_state["mml_approved"]==1)],
        [2,1], default=0).astype(int)
    label_map = {0:"No medical/adult-use", 1:"Medical only", 2:"Adult-use (any)"}
    policy_state["policy_group_label"] = policy_state["policy_group"].map(label_map)
    policy_state.to_csv(policy_min_csv, index=False)

# ── load full policy for switch-year annotation ───────────────────────────────
pol_full = pd.read_excel(policy_xlsx, sheet_name="Full Data")
pol_full.columns = [_snake_case(c) for c in pol_full.columns]
pol_full["statefp"] = _std_statefp_from_policy_fips(pol_full["fips"])
pol_full["year"]    = pd.to_numeric(pol_full["year"], errors="coerce")
for c in ["mml_approved","rec_cann_approved"]:
    if c in pol_full.columns:
        pol_full[c] = pd.to_numeric(pol_full[c], errors="coerce").fillna(0).astype(int)

# ── load CDC overall data ─────────────────────────────────────────────────────
overall = pd.read_csv(overall_csv, dtype=str)
overall.columns = [_snake_case(c) for c in overall.columns]
overall["fips"]    = _std_county_fips(overall["fips"])
overall["statefp"] = overall["fips"].str[:2]
for c in ["year","deaths","population"]:
    overall[c] = pd.to_numeric(overall[c], errors="coerce")

# Join policy group onto county rows
overall = overall.merge(
    policy_state[["statefp","policy_group","policy_group_label"]],
    on="statefp", how="left"
)
overall["policy_group"]       = overall["policy_group"].fillna(0).astype(int)
overall["policy_group_label"] = overall["policy_group_label"].fillna("No medical/adult-use")

# ── build per-substance per-county totals (deaths + population) ───────────────
# This is the ranking table: total deaths 2018-2023 summed per county per drug
county_drug_totals = (
    overall.groupby(["fips","county","statefp","policy_group_label","mcd_type"],
                    as_index=False)
           .agg(total_deaths=("deaths","sum"), total_pop=("population","sum"))
)
# Keep only counties that have at least 1 death for this drug
# (avoids selecting truly-zero counties as "lowest")
county_drug_totals = county_drug_totals.loc[county_drug_totals["total_deaths"] > 0].copy()

# ── helper: policy switch year ────────────────────────────────────────────────
def _policy_switch_year(statefp, col):
    tmp = pol_full.loc[pol_full["statefp"] == statefp, ["year", col]].sort_values("year")
    switched = tmp.loc[tmp[col] == 1, "year"]
    return float(switched.min()) if not switched.empty else None

# ── helper: get county time-series for one drug ───────────────────────────────
def _county_ts(fips, mcd_type):
    dfc = overall.loc[(overall["fips"] == fips) & (overall["mcd_type"] == mcd_type)].copy()
    if dfc.empty:
        return pd.DataFrame(columns=["year","rate_per_100k"])
    ts = (dfc.groupby("year", as_index=False)
             .agg(deaths=("deaths","sum"), population=("population","sum")))
    ts["rate_per_100k"] = _compute_rate(ts)
    return ts[["year","rate_per_100k"]]

# ── helper: build one trend chart for a single county ─────────────────────────
def _single_county_chart(fips, county_name, mcd_type, switch_year,
                         burden_label, policy_label):
    ts = _county_ts(fips, mcd_type)

    title_line1 = f"{burden_label} burden — {county_name} ({fips})"
    title_line2 = f"Policy: {policy_label}"

    if ts.empty or ts["rate_per_100k"].isna().all():
        # Return empty placeholder with informative title
        placeholder = pd.DataFrame({"year":[2018,2023], "rate_per_100k":[0,0]})
        return (alt.Chart(placeholder)
                   .mark_line(opacity=0)
                   .encode(x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
                           y=alt.Y("rate_per_100k:Q", title="Rate per 100k"))
                   .properties(width=380, height=260,
                               title=alt.TitleParams([title_line1, title_line2,
                                                       "(no data for this substance)"],
                                                      fontSize=12)))

    base = alt.Chart(ts).encode(
        x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
        y=alt.Y("rate_per_100k:Q", title="Rate per 100k (county-year)",
                scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("year:Q", format="d"),
                 alt.Tooltip("rate_per_100k:Q", format=".2f", title="Rate/100k")]
    )
    chart = base.mark_line(strokeWidth=2.5, color="#2166ac") + \
            base.mark_point(filled=True, size=60, color="#2166ac")

    if switch_year is not None:
        rule_df = pd.DataFrame({"x": [float(switch_year)]})
        rule = (alt.Chart(rule_df)
                   .mark_rule(color="red", strokeWidth=2, strokeDash=[4,3])
                   .encode(x=alt.X("x:Q")))
        chart = chart + rule

    return chart.properties(
        width=380, height=260,
        title=alt.TitleParams([title_line1, title_line2], fontSize=12)
    )

# ── main loop: 6 drugs × 3 policy groups × 2 (high/low) = 36 charts ──────────
SUBSTANCES = [
    "T401_heroin", "T402_natsemi", "T403_methadone",
    "T404_synthopioids", "T405_cocaine", "T436_psychostim"
]

POLICY_GROUPS = [
    "Adult-use (any)",
    "Medical only",
    "No medical/adult-use"
]

POLICY_SLUG = {
    "Adult-use (any)":      "adultuse",
    "Medical only":         "medicalonly",
    "No medical/adult-use": "nopolicy"
}

generated = []

for mcd in SUBSTANCES:
    drug_totals = county_drug_totals.loc[county_drug_totals["mcd_type"] == mcd].copy()

    for pg_label in POLICY_GROUPS:
        pg_slug    = POLICY_SLUG[pg_label]
        pg_totals  = drug_totals.loc[drug_totals["policy_group_label"] == pg_label].copy()

        if pg_totals.empty:
            print(f"  ⚠ No counties with deaths: {mcd} / {pg_label} — skipping")
            continue

        # Select highest and lowest county by total_deaths
        pg_sorted   = pg_totals.sort_values("total_deaths", ascending=False)
        row_high    = pg_sorted.iloc[0]
        row_low     = pg_sorted.iloc[-1]

        for burden_label, row in [("Highest", row_high), ("Lowest", row_low)]:
            fips        = row["fips"]
            county_name = str(row["county"])
            statefp     = row["statefp"]
            total_d     = int(row["total_deaths"])

            # Switch year: adult-use annotation for all policy groups
            # (Medical-only & No-policy states won't have one, which is fine)
            switch_year = _policy_switch_year(statefp, "rec_cann_approved")

            chart = _single_county_chart(
                fips, county_name, mcd, switch_year,
                burden_label=f"{burden_label} ({total_d} deaths)",
                policy_label=pg_label
            )

            # Full chart with top-level title
            full_chart = chart.properties().configure_title(
                fontSize=13, fontWeight="bold", anchor="start"
            )

            slug = burden_label.lower()
            fname = f"fig6_{mcd}_{pg_slug}_{slug}_burden"
            _safe_save(full_chart, out_dir / fname)

            generated.append({
                "file": fname + ".html",
                "drug": mcd,
                "policy": pg_label,
                "burden": burden_label,
                "county": county_name,
                "fips": fips,
                "total_deaths": total_d
            })

            print(f"  ✓ {fname}  →  {county_name} ({fips}), deaths={total_d}")

# ── summary table ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(generated)
summary_path = out_dir / "fig6_county_selection_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n{'='*70}")
print(f"Generated {len(generated)} charts  →  {out_dir}")
print(f"County selection log  →  {summary_path}")
print(summary_df.to_string(index=False))