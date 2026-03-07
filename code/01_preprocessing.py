import pandas as pd 
import numpy as np
from pathlib import Path
import re

# Please change data path as per your folder organization before re-running this file 
data_path = '/Users/koniks/Desktop/GitHub Folder/pubhealth-dataviz/data'
cdc_raw = data_path + '/raw/mcd-cdc'
derived_dir = Path(data_path) / "derived"

# =====================================================================================
# Helpers (standardize entries, test variable types, derive death type from file name)
# =====================================================================================

def _snake_case(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("%", "pct").replace("/", "_").replace("-", "_")
    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_snake_case(c) for c in df.columns]
    df = df.rename(columns={
        "county_code": "fips",
        "crude_rate_lower_95pct_confidence_interval": "crude_rate_ci_lower",
        "crude_rate_upper_95pct_confidence_interval": "crude_rate_ci_upper",
    })
    return df

def _standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize types across WONDER extracts:
    - Keep text fields as strings (trimmed)
    - Year -> numeric (Int64)
    - Numeric measures -> numeric via to_numeric
    - FIPS -> 5-digit string (handles '38' vs '0038', etc.)
    """
    df = _standardize_columns(df)

    # Trim whitespace for object-like columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # Ensure FIPS exists and is standardized (5-digit string)
    if "fips" in df.columns:
        df["fips"] = (
            df["fips"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )

    # Year to numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Standard numeric columns if present
    numeric_cols = [
        "deaths", "population", "crude_rate",
        "crude_rate_ci_lower", "crude_rate_ci_upper"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") # this code line also removes 'not available' values from Population, Crude Rate and Confidence Interval columns in the 'Overall' file

    return df

def _infer_mcd_type_from_filename(filename: str) -> str:
    """
    Derive the multiple-cause-of-death type from your filename convention:
      od_T401_heroin_county_year_2018_2023.csv -> T401_heroin
      od_T402_natsemi_county_sex.csv           -> T402_natsemi
      od_T436_psychostim_county_race.csv       -> T436_psychostim
    """
    stem = Path(filename).stem  # drop .csv
    if stem.startswith("od_"):
        stem = stem[3:]
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem

# ==================================================================================================================
# A. Cleaning each file separately, excluding age which is cleaned separately at the end of the preprocessing file  
# ==================================================================================================================
# Remove extra rows at the end detailing dataset information
def _remove_footer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify where CDC WONDER footer/metadata rows begin and remove them.

    Heuristic:
    - Find the first row where County is blank OR County Code is non-numeric/blank.
    - Keep everything above that row.
    """
    df = df.copy()
    df = _standardize_columns(df)

    if "county" not in df.columns or "fips" not in df.columns:
        return df

    county_blank = df["county"].astype(str).str.strip().eq("") | df["county"].isna()
    fips_raw = df["fips"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    fips_bad = fips_raw.eq("") | ~fips_raw.str.match(r"^\d+$", na=False)

    footer_start_idx = df.index[(county_blank | fips_bad)]
    if len(footer_start_idx) == 0:
        return df
    
    first_footer = int(footer_start_idx.min())
    return df.loc[df.index < first_footer].copy()

# =============================================================================
# Data Aggregation
# Goal: each folder -> one aggregated DF with mcd_type column
# =============================================================================

# 1. Overall (overall_county_2018_2023)
overall_folder = Path(cdc_raw) / "overall_county_2018_2023"
overall_paths = sorted(overall_folder.glob("*.csv"))
if len(overall_paths) == 0:
    raise FileNotFoundError(f"No CSV files found in: {overall_folder}")

overall_frames = []
for p in overall_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)
    overall_frames.append(df)

overall_county_2018_2023_all = pd.concat(overall_frames, ignore_index=True)

# 2. Urbanization (county_urbanization2023)
urban_folder = Path(cdc_raw) / "county_urbanization2023"
urban_paths = sorted(urban_folder.glob("*.csv"))
if len(urban_paths) == 0:
    raise FileNotFoundError(f"No CSV files found in: {urban_folder}")

urban_frames = []
for p in urban_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)
    urban_frames.append(df)

county_urbanization_all = pd.concat(urban_frames, ignore_index=True)

# 3. Sex (county_sex_2018_2023)
sex_folder = Path(cdc_raw) / "county_sex_2018_2023"
sex_paths = sorted(sex_folder.glob("*.csv"))
if len(sex_paths) == 0:
    raise FileNotFoundError(f"No CSV files found in: {sex_folder}")

sex_frames = []
for p in sex_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)
    sex_frames.append(df)

county_sex_all = pd.concat(sex_frames, ignore_index=True)

# 4. Race (county_race_2018_2023)
race_folder = Path(cdc_raw) / "county_race_2018_2023"
race_paths = sorted(race_folder.glob("*.csv"))
if len(race_paths) == 0:
    raise FileNotFoundError(f"No CSV files found in: {race_folder}")

race_frames = []
for p in race_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)
    race_frames.append(df)

county_race_all = pd.concat(race_frames, ignore_index=True)

# 5. Education (county_education_2018_2023)
educ_folder = Path(cdc_raw) / "county_education_2018_2023"
educ_paths = sorted(educ_folder.glob("*.csv"))
if len(educ_paths) == 0:
    raise FileNotFoundError(f"No CSV files found in: {educ_folder}")

educ_frames = []
for p in educ_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)
    educ_frames.append(df)

county_education_all = pd.concat(educ_frames, ignore_index=True)

# =============================================================================
# C. Recoding 'Not Available' as missing=. , counting number of missing rows 
# 2. Urbanization
# 3. Sex
# 4. Race
# 5. Education
# =============================================================================
missing_urbanization = county_urbanization_all[["population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]].isna().sum()
missing_sex = county_sex_all[["population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]].isna().sum()
missing_race = county_race_all[["population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]].isna().sum()
missing_education = county_education_all[["population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]].isna().sum()

# =============================================================================
# D. Reporting on missing columns for each county
# 1. Overall
# 2. Urbanization
# 3. Sex
# 4. Race
# 5. Education
# =============================================================================
value_cols = ["deaths", "population", "crude_rate", "crude_rate_ci_lower", "crude_rate_ci_upper"]

missing_by_county_overall = overall_county_2018_2023_all.groupby(["fips", "county"])[value_cols].apply(lambda g: g.isna().sum()).reset_index()
missing_by_county_urbanization = county_urbanization_all.groupby(["fips", "county"])[value_cols].apply(lambda g: g.isna().sum()).reset_index()
missing_by_county_sex = county_sex_all.groupby(["fips", "county"])[value_cols].apply(lambda g: g.isna().sum()).reset_index()
missing_by_county_race = county_race_all.groupby(["fips", "county"])[value_cols].apply(lambda g: g.isna().sum()).reset_index()
missing_by_county_education = county_education_all.groupby(["fips", "county"])[value_cols].apply(lambda g: g.isna().sum()).reset_index()

print(missing_by_county_overall, missing_by_county_urbanization, missing_by_county_sex, missing_by_county_race, missing_by_county_education)

# The missing columns showed a very positive result. Missingness is very low, and concentrated in population/crude rate, not deaths. The safest 
# decision is to filter these values in our plotting stage out since they were likely just suppressed during the data shopping step or not available in reality.
# This should also prevent data loss from the other columns, from whom we do have meaningful data available (and we might not even use the population column for most visualizations!)

# =============================================================================
# E. Age file - specific cleaning steps 
# =============================================================================

# Data Aggregation 
# For county_age_2018_2023 files, they had to be downloaded in smaller chunks to overcome the 75,000 observation limit of CDC WONDER
# This step aggregates the data 

# Add all _natsemi_ files together and then order by year 

# 6. Age (county_age_2018_2023) with subfolders 2023/ and 2024/
age_folder = Path(cdc_raw) / "county_age_2018_2023"

# Root files 
age_root_paths = sorted(age_folder.glob("*.csv"))
if len(age_root_paths) == 0:
    raise FileNotFoundError(f"No root-level age CSV files found in: {age_folder}")

age_root_frames = []
for p in age_root_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)

    # If filename ends with a 4-digit year (natsemi root files), set year from filename
    last_token = p.stem.split("_")[-1]
    if last_token.isdigit() and len(last_token) == 4:
        df["year"] = int(last_token)

    # Drop year_code if present
    if "year_code" in df.columns:
        df = df.drop(columns=["year_code"])

    age_root_frames.append(df)

county_age_root_all = pd.concat(age_root_frames, ignore_index=True)

# 2023 chunk files
age_2023_folder = age_folder / "2023"
age_2023_paths = sorted(age_2023_folder.glob("*.csv"))
if len(age_2023_paths) == 0:
    raise FileNotFoundError(f"No 2023 age CSV files found in: {age_2023_folder}")

age_2023_frames = []
for p in age_2023_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)

    # year from folder name
    df["year"] = 2023

    if "year_code" in df.columns:
        df = df.drop(columns=["year_code"])

    age_2023_frames.append(df)

county_age_2023_all = pd.concat(age_2023_frames, ignore_index=True)

# 2024 chunk files
age_2024_folder = age_folder / "2024"
age_2024_paths = sorted(age_2024_folder.glob("*.csv"))
if len(age_2024_paths) == 0:
    raise FileNotFoundError(f"No 2024 age CSV files found in: {age_2024_folder}")

age_2024_frames = []
for p in age_2024_paths:
    df = pd.read_csv(p, dtype=str)
    df = _remove_footer_rows(df)
    df = _standardize_types(df)
    df["mcd_type"] = _infer_mcd_type_from_filename(p.name)

    # year from folder name 
    df["year"] = 2024

    if "year_code" in df.columns:
        df = df.drop(columns=["year_code"])

    age_2024_frames.append(df)

county_age_2024_all = pd.concat(age_2024_frames, ignore_index=True)

# combine all years and validate no missing year

county_age_all = pd.concat(
    [county_age_root_all, county_age_2023_all, county_age_2024_all],
    ignore_index=True
)

# =============================================================================
# F. Write each aggregated folder output to data/derived (uncomment after checking)
# =============================================================================
derived_dir.mkdir(parents=True, exist_ok=True)

overall_county_2018_2023_all.to_csv(derived_dir / "overall_county_2018_2023_all_mcd_types.csv", index=False)
county_urbanization_all.to_csv(derived_dir / "county_urbanization2023_all_mcd_types.csv", index=False)
county_sex_all.to_csv(derived_dir / "county_sex_2018_2023_all_mcd_types.csv", index=False)
county_race_all.to_csv(derived_dir / "county_race_2018_2023_all_mcd_types.csv", index=False)
county_education_all.to_csv(derived_dir / "county_education_2018_2023_all_mcd_types.csv", index=False)
county_age_all.to_csv(derived_dir / "county_age_2018_2024_all_mcd_types.csv", index=False)