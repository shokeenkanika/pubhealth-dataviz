# ==============================
# spatial_heterogeneityfigures.py
# ==============================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Please change paths as per your folder organization before re-running this file
project_root = Path('/Users/koniks/Desktop/GitHub Folder/pubhealth-dataviz')
data_path = project_root / "data"
cdc_derived = data_path / "derived"
dataverse_raw = data_path / "raw" / "dataverse_files"
fig_dir = project_root / "outputs" / "figures" / "heterogeneity_maps"

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
# 1) Load policy dataset and build policy_group (same definition as policy script)
# -----------------------------------------------------------------------------
policy_xlsx = dataverse_raw / "Cannabis Policies Full Dataset 81425.xlsx"
policy = pd.read_excel(policy_xlsx, sheet_name="Full Data")
policy.columns = [_snake_case(c) for c in policy.columns]

policy["statefp"] = _standardize_fips(policy["fips"]).str[:2]
policy["year"] = pd.to_numeric(policy["year"], errors="coerce").astype("Int64")

for col in ["mml_approved", "rec_cann_approved"]:
    policy[col] = pd.to_numeric(policy[col], errors="coerce").fillna(0).astype(int)

policy_18_23 = policy.loc[policy["year"].between(2018, 2023)].copy()
policy_state = (
    policy_18_23.groupby("statefp", as_index=False)
               .agg(mml_approved=("mml_approved", "max"),
                    rec_cann_approved=("rec_cann_approved", "max"))
)

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

# -----------------------------------------------------------------------------
# 2) Load CDC heterogeneity files (urbanization / education / sex)
#    (Age will be added later)
# -----------------------------------------------------------------------------
urban_csv = cdc_derived / "county_urbanization2023_all_mcd_types.csv"
sex_csv = cdc_derived / "county_sex_2018_2023_all_mcd_types.csv"
educ_csv = cdc_derived / "county_education_2018_2023_all_mcd_types.csv"

urban = pd.read_csv(urban_csv, dtype=str)
sex = pd.read_csv(sex_csv, dtype=str)
educ = pd.read_csv(educ_csv, dtype=str)

for df in [urban, sex, educ]:
    df.columns = [_snake_case(c) for c in df.columns]
    df["fips"] = _standardize_fips(df["fips"])
    df["statefp"] = df["fips"].str[:2]
    for c in ["deaths", "population", "year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------------------------------------------------------
# 3) Aggregate across time (ignore year) and compute rates
# -----------------------------------------------------------------------------
def _agg_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, as_index=False)
          .agg(deaths=("deaths", "sum"), population=("population", "sum"))
    )
    out["rate_per_100k"] = (out["deaths"] / out["population"]) * 100_000.0
    out.loc[out["population"].isna() | (out["population"] == 0), "rate_per_100k"] = np.nan
    return out

# Join policy group to each dataset (state-level)
urban = urban.merge(policy_state[["statefp", "policy_group", "policy_group_label"]], on="statefp", how="left")
sex = sex.merge(policy_state[["statefp", "policy_group", "policy_group_label"]], on="statefp", how="left")
educ = educ.merge(policy_state[["statefp", "policy_group", "policy_group_label"]], on="statefp", how="left")

# -----------------------------------------------------------------------------
# 4) Heterogeneous effects figures (policy vs non-policy within strata)
# -----------------------------------------------------------------------------
# Urbanization: policy comparisons separately for urbanization categories
# (column names from your files: "2023_urbanization" becomes "2023_urbanization" after snake_case)
urban_col = "2023_urbanization" if "2023_urbanization" in urban.columns else "urbanization"
urban_agg = _agg_rate(urban, ["policy_group_label", urban_col])
urban_pivot = urban_agg.pivot(index=urban_col, columns="policy_group_label", values="rate_per_100k")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
urban_pivot.plot(kind="bar", ax=ax)
ax.set_title("Heterogeneous effects: policy group differences by 2023 urbanization (rate per 100k)")
ax.set_xlabel("Urbanization category")
ax.set_ylabel("Rate per 100k (2018–2023 aggregated)")
plt.tight_layout()
plt.savefig(fig_dir / "HET_urbanization_policy_rate.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Education: compare gradients within policy group
educ_col = "education" if "education" in educ.columns else "education_level"
educ_agg = _agg_rate(educ, ["policy_group_label", educ_col])
educ_pivot = educ_agg.pivot(index=educ_col, columns="policy_group_label", values="rate_per_100k")

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
educ_pivot.plot(kind="bar", ax=ax)
ax.set_title("Heterogeneous effects: policy group differences by education (rate per 100k)")
ax.set_xlabel("Education category")
ax.set_ylabel("Rate per 100k (2018–2023 aggregated)")
plt.tight_layout()
plt.savefig(fig_dir / "HET_education_policy_rate.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Sex: show whether policy-group gap is bigger among males vs females
sex_col = "sex" if "sex" in sex.columns else "gender"
sex_agg = _agg_rate(sex, ["policy_group_label", sex_col])
sex_pivot = sex_agg.pivot(index=sex_col, columns="policy_group_label", values="rate_per_100k")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sex_pivot.plot(kind="bar", ax=ax)
ax.set_title("Heterogeneous effects: policy group differences by sex (rate per 100k)")
ax.set_xlabel("Sex")
ax.set_ylabel("Rate per 100k (2018–2023 aggregated)")
plt.tight_layout()
plt.savefig(fig_dir / "HET_sex_policy_rate.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Age: later (intentionally not included yet)