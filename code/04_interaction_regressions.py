# -----------------------------------------------------------------------------
# This script is part of the replication workflow. It assumes that
# code/00_clean_data.py has already converted original files in raw_data/ into
# clean_data/. This script reads cleaned inputs only; it does not read raw_data/.
# -----------------------------------------------------------------------------

"""Estimate the selected-sample heterogeneous-effects interaction table.

This script produces the paper's descriptive interaction table. It asks whether
the relationship between log calibrated subsistence intensity and log SNAP
participation rate differs by historical crop-share regimes and fieldwork
suitability. These are descriptive interaction regressions, not the main IV
identification design.

Published output:
  * outputs/regression/interaction_table.tex
"""
# -----------------------------------------------------------------------------
# READER GUIDE
# -----------------------------------------------------------------------------
# This file estimates descriptive interaction models. These are not the causal
# IV design; they show whether the baseline association differs across historical
# crop-share regimes and fieldwork suitability. It writes the interaction table
# used in the paper.
# -----------------------------------------------------------------------------

# ============================================================
# MATCHING-SCORE SELECTED RELIABILITY-WEIGHTED INTERACTION MODELS
# SEPARATE CROP-SHARE HETEROGENEITY
# WITH NO-FE AND REGION-FE OLS/WLS/WLS-SQUARED VARIANTS
# + THRESHOLD ROBUSTNESS
#
# Purpose:
#   Estimate descriptive heterogeneous-effects regressions showing whether the
#   relationship between calibrated subsistence intensity and SNAP participation
#   differs by historical agricultural regime.
#
# Main interaction design:
#   Outcome:        log SNAP participation rate
#   Main regressor: log(optimized calibrated subsistence parameter)
#   Heterogeneity variables:
#       historical cotton share
#       historical corn share
#       historical soybean share
#       fieldwork suitability
#
# Table layout:
#   Each crop/condition is reported as a vertical panel.
#   Columns report:
#       No Region FE: OLS, WLS, WLS Sq.
#       Region FE:    OLS, WLS, WLS Sq.
#
# Important:
#   Interaction models use crop shares, not log crop acreage.
#   This is intentional because these models describe relative historical
#   agricultural regimes rather than serving as the IV design.
#
# Main selected sample:
#   matching_score_percent > 90
#
# Robustness thresholds:
#   matching_score_percent > 85
#   matching_score_percent > 90
#   matching_score_percent > 95
# ============================================================

# Anaconda Prompt:
# pip install pandas numpy statsmodels openpyxl

import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# FILE PATHS
# ============================================================

crop_files = {
    "soybean": str(ROOT / "clean_data" / "cash_crop_data" / "soybean.csv"),
    "corn": str(ROOT / "clean_data" / "cash_crop_data" / "corn.csv"),
    "cotton": str(ROOT / "clean_data" / "cash_crop_data" / "cotton.csv"),
}

abar_path = str(ROOT / "derived" / "calibration" / "calibrated_state_abar_optimized_alpha_raw_productivity.csv")
quality_path = str(ROOT / "derived" / "calibration" / "soft_dtw_matching_quality_by_state_raw_productivity.csv")
suitability_path = str(ROOT / "clean_data" / "cash_crop_data" / "suitability_clean.csv")
snap_path = str(ROOT / "clean_data" / "snap" / "snap-persons-4.xlsx")
population_path = str(ROOT / "clean_data" / "snap" / "NST-EST2025-POP.xlsx")

output_dir = str(ROOT / "derived" / "interaction")
os.makedirs(output_dir, exist_ok=True)
MANUSCRIPT_REGRESSION_DIR = ROOT / "outputs" / "regression"
MANUSCRIPT_REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_REGRESSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METHOD VALIDATION
# ============================================================

def assert_baseline_calibration_method():
    """Stop early if downstream regressions are not using the intended calibration.

    The paper's current method requires:
      1. raw USDA labor productivity, not within-state 1960 normalization;
      2. baseline early calibration moments from 1960 and 1970;
      3. the baseline derived calibration files produced by 02_calibration_model.py.
    """
    summary_path = ROOT / "derived" / "calibration" / "calibration_summary_optimized_alpha_raw_productivity.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing baseline calibration summary: {summary_path}. Run 02_calibration_model.py first."
        )
    summary = pd.read_csv(summary_path)
    sdict = dict(zip(summary["statistic"], summary["value"]))
    prod = str(sdict.get("productivity_variable", ""))
    years = str(sdict.get("early_years_for_abar", ""))
    if prod != "Aa_labor_productivity_index":
        raise ValueError(f"Wrong productivity variable for regressions: {prod}")
    if years.replace(" ", "") != "1960,1970":
        raise ValueError(f"Regressions must use baseline 1960+1970 calibration, but found: {years}")
    if "1960_state1" in str(abar_path) or "1960_state1" in str(quality_path):
        raise ValueError("Regression paths refer to the old normalized calibration. Update paths before running.")

# ============================================================
# SETTINGS
# ============================================================

YVAR = "log_snap_rate"
ABAR_RAW = "abar_state"
XVAR = "log_abar_state"
MATCH = "matching_score_percent"

CROPS = ["cotton", "corn", "soybean"]
CROP_START_YEAR = 1926
CROP_END_YEAR = 1940

CROP_SHARE_VARS = {
    "cotton": "share_cotton",
    "corn": "share_corn",
    "soybean": "share_soybean",
}

SUIT = "suitability"
START_MONTH = 4
END_MONTH = 10

MAIN_SELECTED_MATCH_THRESHOLD = 90.0
ROBUSTNESS_THRESHOLDS = [85.0, 90.0, 95.0]

WEIGHT_SPECS = [
    ("OLS", None, "No"),
    ("WLS", "match_weight", "$Match/100$"),
    ("WLS Sq.", "match_weight_sq", "$(Match/100)^2$"),
]

FE_SPECS = [
    ("No Region FE", []),
    ("Region FE", "REGION_FE"),
]


# Baseline specification (no interaction)
BASELINE_SPEC = ("Baseline", None, None)


SEPARATE_INTERACTION_SPECS = [
    ("Cotton Share Interaction", "Cotton share", "share_cotton", "log_abar_x_share_cotton"),
    ("Corn Share Interaction", "Corn share", "share_corn", "log_abar_x_share_corn"),
    ("Soybean Share Interaction", "Soybean share", "share_soybean", "log_abar_x_share_soybean"),
    ("Suitability Interaction", "Suitability", SUIT, "log_abar_x_suitability"),
]

# ============================================================
# STATE MAPS
# ============================================================

state_to_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

abbr_to_region = {
    "AL": "South", "AR": "South", "DE": "South", "FL": "South",
    "GA": "South", "KY": "South", "LA": "South", "MD": "South",
    "MS": "South", "NC": "South", "OK": "South", "SC": "South",
    "TN": "South", "TX": "South", "VA": "South", "WV": "South",

    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest",
    "KS": "Midwest", "MI": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest",
    "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",

    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
    "NH": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast", "RI": "Northeast", "VT": "Northeast",

    "AK": "West", "AZ": "West", "CA": "West", "CO": "West",
    "HI": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "OR": "West", "UT": "West", "WA": "West",
    "WY": "West"
}

# ============================================================
# HELPERS
# ============================================================

def safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name.strip("._")[:80]


def short_code(value):
    """Compact code for filenames to avoid Windows path-length errors."""
    mapping = {
        "Baseline": "base",
        "Cotton Share Interaction": "cotton",
        "Corn Share Interaction": "corn",
        "Soybean Share Interaction": "soy",
        "Suitability Interaction": "suit",
        "No Region FE": "nofe",
        "Region FE": "fe",
        "OLS": "ols",
        "WLS": "wls",
        "WLS Sq.": "wlssq",
    }
    return mapping.get(str(value), safe_filename(value).lower()[:20])


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def coef_se(coef, se, p):
    if pd.isna(coef) or pd.isna(se):
        return ""
    st = stars(p)
    if st:
        return f"${coef:.3f}^{{{st}}}$\n$({se:.3f})$"
    return f"${coef:.3f}$\n$({se:.3f})$"


def clean_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def load_state_population_2025(path):
    """Load July 1, 2025 state resident population from Census NST-EST2025-POP.

    Expected file:
      Annual Estimates of the Resident Population for the United States, Regions,
      States, District of Columbia and Puerto Rico: April 1, 2020 to July 1, 2025.

    The Census table uses row 3/4 style headers and puts states in the first column
    with leading dots (e.g., ".Alabama"). This loader is intentionally defensive so
    it keeps working if the downloaded workbook has a short title/header block.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)

    # Find the row containing the year headers 2020--2025.
    header_row_idx = None
    for i in range(min(15, len(raw))):
        vals = set(pd.to_numeric(raw.iloc[i], errors="coerce").dropna().astype(int).tolist())
        if 2025 in vals and 2024 in vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Could not find the Census year-header row containing 2025 in population file.")

    # The geography column is column 0. The 2025 population column is where the header row equals 2025.
    year_row = pd.to_numeric(raw.iloc[header_row_idx], errors="coerce")
    pop_cols = [j for j, v in year_row.items() if pd.notna(v) and int(v) == 2025]
    if not pop_cols:
        raise ValueError("Could not find a 2025 population column in the Census population file.")
    pop_col = pop_cols[0]

    pop = raw.iloc[header_row_idx + 1:, [0, pop_col]].copy()
    pop.columns = ["state_name_raw", "population_2025"]

    pop["state_name"] = (
        pop["state_name_raw"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.title()
    )
    pop["population_2025"] = pd.to_numeric(pop["population_2025"], errors="coerce")
    pop["state"] = pop["state_name"].map(state_to_abbr)

    pop = pop.dropna(subset=["state", "population_2025"]).copy()
    pop = pop[pop["population_2025"] > 0].copy()

    # Keep one row per state.
    pop = pop[["state", "state_name", "population_2025"]].drop_duplicates("state", keep="first")
    return pop


def filter_weights(d, weight_col):
    if weight_col is None:
        return d
    return d[np.isfinite(d[weight_col]) & (d[weight_col] > 0)].copy()


def full_rank_controls(d, controls, base_cols=None):
    """Keep only controls that add independent variation beyond base regressors."""
    base_cols = [] if base_cols is None else list(base_cols)
    controls = [c for c in controls if c in d.columns]
    kept = []

    X_parts = [np.ones((len(d), 1))]
    if base_cols:
        X_parts.append(d[base_cols].to_numpy(dtype=float))

    X = np.column_stack(X_parts)
    current_rank = np.linalg.matrix_rank(X)

    for col in controls:
        trial = np.column_stack([X, d[[col]].to_numpy(dtype=float)])
        trial_rank = np.linalg.matrix_rank(trial)
        if trial_rank > current_rank:
            kept.append(col)
            X = trial
            current_rank = trial_rank

    dropped = [c for c in controls if c not in kept]
    return kept, dropped


def run_ols_or_wls(y, xvars, controls, data, weight_col=None):
    needed = [y] + list(xvars) + list(controls)
    if weight_col is not None:
        needed.append(weight_col)

    d = data.dropna(subset=needed).copy()
    d = clean_numeric(d, needed).dropna(subset=needed)
    d = filter_weights(d, weight_col)

    if len(d) < 8:
        raise ValueError(f"Too few observations: {len(d)}")

    controls_kept, controls_dropped = full_rank_controls(
        d=d,
        controls=controls,
        base_cols=xvars
    )

    used_xvars = list(xvars) + controls_kept
    X = sm.add_constant(d[used_xvars], has_constant="add")

    if np.linalg.matrix_rank(X.to_numpy(dtype=float)) < X.shape[1]:
        raise ValueError("Regressor matrix is not full rank.")

    if weight_col is None:
        model = sm.OLS(d[y], X).fit(cov_type="HC1")
    else:
        model = sm.WLS(d[y], X, weights=d[weight_col]).fit(cov_type="HC1")

    model.used_controls = controls_kept
    model.dropped_controls = controls_dropped
    return model, d


assert_baseline_calibration_method()

# ============================================================
# LOAD DATA
# ============================================================

cleaned_crops = []

for crop_name, path in crop_files.items():
    temp = pd.read_csv(path)

    required_cols = ["Geo Level", "Period", "Domain", "Value", "State", "Year"]
    missing = [c for c in required_cols if c not in temp.columns]
    if missing:
        raise ValueError(f"{crop_name} crop file missing required columns: {missing}")

    temp = temp[
        (temp["Geo Level"] == "STATE") &
        (temp["Period"] == "YEAR") &
        (temp["Domain"] == "TOTAL")
    ].copy()

    temp["value"] = (
        temp["Value"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(D)", "", regex=False)
        .str.replace("(Z)", "0", regex=False)
        .str.strip()
    )
    temp["value"] = pd.to_numeric(temp["value"], errors="coerce")
    temp["state_name"] = temp["State"].astype(str).str.title().str.strip()
    temp["state"] = temp["state_name"].map(state_to_abbr)
    temp["year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp["crop"] = crop_name

    temp = temp[["state", "state_name", "year", "crop", "value"]]
    cleaned_crops.append(temp)

crop_panel = pd.concat(cleaned_crops, ignore_index=True)
crop_panel = crop_panel.dropna(subset=["state", "year", "value"])
crop_panel["year"] = crop_panel["year"].astype(int)

crop_panel = crop_panel[
    (crop_panel["year"] >= CROP_START_YEAR) &
    (crop_panel["year"] <= CROP_END_YEAR)
].copy()

crop_state = crop_panel.groupby(["state", "crop"], as_index=False)["value"].mean()
crop_wide = crop_state.pivot(index="state", columns="crop", values="value").reset_index()

for crop in CROPS:
    if crop not in crop_wide.columns:
        crop_wide[crop] = 0.0

crop_wide[CROPS] = crop_wide[CROPS].fillna(0.0)
crop_wide["cash_crop_sum"] = crop_wide[CROPS].sum(axis=1)

for crop in CROPS:
    crop_wide[f"{crop}_acres_1926_1940"] = crop_wide[crop]
    crop_wide[f"share_{crop}"] = np.where(
        crop_wide["cash_crop_sum"] > 0,
        crop_wide[crop] / crop_wide["cash_crop_sum"],
        np.nan
    )

crop_keep_cols = ["state"] + [f"{crop}_acres_1926_1940" for crop in CROPS] + [f"share_{crop}" for crop in CROPS]
crop_wide = crop_wide[crop_keep_cols].copy()

# Abar and matching quality
abar = pd.read_csv(abar_path)
quality = pd.read_csv(quality_path)

abar["state"] = abar["state"].astype(str).str.upper().str.strip()
quality["state"] = quality["state"].astype(str).str.upper().str.strip()

if ABAR_RAW not in abar.columns:
    raise ValueError(
        f"Expected column {ABAR_RAW!r} in optimized calibration file. "
        "Check that abar_path points to calibrated_state_abar_optimized_alpha_raw_productivity.csv."
    )

if MATCH not in quality.columns:
    raise ValueError(f"Expected column {MATCH!r} in matching-quality file.")

df_abar = abar.merge(quality[["state", MATCH]], on="state", how="left")
df_abar = clean_numeric(df_abar, [ABAR_RAW, MATCH])
df_abar = df_abar[df_abar[ABAR_RAW] > 0].copy()
df_abar[XVAR] = np.log(df_abar[ABAR_RAW])

# Suitability
suit = pd.read_csv(suitability_path)
suit["state_name"] = suit["state"].astype(str).str.title().str.strip()
suit["state"] = suit["state_name"].map(state_to_abbr)
suit["week_ending"] = pd.to_datetime(suit["week_ending"], errors="coerce")
suit["month"] = suit["week_ending"].dt.month
suit[SUIT] = pd.to_numeric(suit[SUIT], errors="coerce")
suit = suit[suit["month"].between(START_MONTH, END_MONTH)].copy()
suit = suit.groupby(["state", "state_name"], as_index=False)[SUIT].mean()
suit = suit.dropna(subset=["state", SUIT])

# SNAP
snap = pd.read_excel(snap_path, sheet_name=0, header=2)
snap = snap.rename(columns={snap.columns[0]: "state_name"})
snap["state_name"] = snap["state_name"].astype(str).str.title().str.strip()
snap["state"] = snap["state_name"].map(state_to_abbr)

possible_cols = snap.columns[1:4]
snap["snap_persons"] = np.nan
snap_used_column = None

for col in possible_cols[::-1]:
    temp = pd.to_numeric(snap[col], errors="coerce")
    if temp.notna().sum() > 20:
        snap["snap_persons"] = temp
        snap_used_column = col
        break

snap = snap.dropna(subset=["state", "snap_persons"]).copy()
snap["snap_persons"] = pd.to_numeric(snap["snap_persons"], errors="coerce")
snap = snap[snap["snap_persons"] > 0].copy()

# Population denominator: Census July 1, 2025 resident population.
population = load_state_population_2025(population_path)
snap = snap.merge(population[["state", "population_2025"]], on="state", how="inner")

snap["snap_rate"] = snap["snap_persons"] / snap["population_2025"]
snap = snap[np.isfinite(snap["snap_rate"]) & (snap["snap_rate"] > 0)].copy()

# Main dependent variable for the paper.
snap[YVAR] = np.log(snap["snap_rate"])

# Keep log SNAP persons as a diagnostic/robustness variable, but do not use it as the main outcome.
snap["log_snap_persons"] = np.log(snap["snap_persons"])

snap = snap[[
    "state", "state_name", "snap_persons", "population_2025",
    "snap_rate", "log_snap_persons", YVAR
]].copy()

print("SNAP column used:", snap_used_column)
print("Population denominator: Census NST-EST2025-POP, July 1, 2025 resident population")

# ============================================================
# MERGE BASE DATASET
# ============================================================

base_reg = (
    snap
    .merge(df_abar, on="state", how="inner")
    .merge(crop_wide, on="state", how="inner")
    .merge(suit[["state", SUIT]], on="state", how="inner")
)

base_reg["region"] = base_reg["state"].map(abbr_to_region)

needed_base = [
    "state", "region", YVAR, XVAR, MATCH, SUIT,
    "share_cotton", "share_corn", "share_soybean"
]
base_reg = base_reg.dropna(subset=needed_base).copy()

base_reg["match_weight"] = base_reg[MATCH] / 100
base_reg["match_weight_sq"] = base_reg["match_weight"] ** 2
base_reg.loc[base_reg["match_weight"] <= 0, "match_weight"] = np.nan
base_reg.loc[base_reg["match_weight_sq"] <= 0, "match_weight_sq"] = np.nan

for crop in CROPS:
    share_var = CROP_SHARE_VARS[crop]
    base_reg[f"log_abar_x_{share_var}"] = base_reg[XVAR] * base_reg[share_var]

base_reg["log_abar_x_suitability"] = base_reg[XVAR] * base_reg[SUIT]

# ============================================================
# ESTIMATION AND TABLE BUILDERS
# ============================================================

def build_region_fe(reg):
    region_dummies = pd.get_dummies(
        reg["region"],
        prefix="region",
        drop_first=True,
        dtype=float
    )
    out = pd.concat([reg.copy(), region_dummies], axis=1)
    return out, list(region_dummies.columns)


def prepare_sample(base, threshold=None):
    reg = base.copy()
    if threshold is not None:
        reg = reg[reg[MATCH] > threshold].copy()

    if reg.empty:
        raise ValueError(f"No observations for threshold {threshold}.")

    reg, region_fe = build_region_fe(reg)

    numeric_cols = [
        YVAR, XVAR, MATCH, SUIT,
        "share_cotton", "share_corn", "share_soybean",
        "log_abar_x_share_cotton",
        "log_abar_x_share_corn",
        "log_abar_x_share_soybean",
        "log_abar_x_suitability",
        "match_weight",
        "match_weight_sq",
    ] + region_fe

    reg = clean_numeric(reg, numeric_cols)
    return reg, region_fe


TABLE_COLS = [
    ("No Region FE", "OLS", None),
    ("No Region FE", "WLS", "match_weight"),
    ("No Region FE", "WLS Sq.", "match_weight_sq"),
    ("Region FE", "OLS", None),
    ("Region FE", "WLS", "match_weight"),
    ("Region FE", "WLS Sq.", "match_weight_sq"),
]


def estimate_one_panel(reg, region_fe, panel_name, main_label, z_var, xz_var, sample_key, sample_dir):
    """Estimate one crop/condition interaction panel across all six table columns."""
    models = {}
    rows = []

    for fe_label, estimator_label, weight_col in TABLE_COLS:
        controls = [] if fe_label == "No Region FE" else region_fe
        xvars = [XVAR, z_var, xz_var]
        col_key = (fe_label, estimator_label)

        try:
            model, used_data = run_ols_or_wls(
                y=YVAR,
                xvars=xvars,
                controls=controls,
                data=reg,
                weight_col=weight_col
            )
        except Exception as e:
            print(f"Skipped {panel_name} | {fe_label} | {estimator_label}: {repr(e)}")
            continue

        models[col_key] = model

        os.makedirs(sample_dir, exist_ok=True)
        summary_name = f"{short_code(sample_key)}_{short_code(panel_name)}_{short_code(fe_label)}_{short_code(estimator_label)}.txt"
        with open(
            os.path.join(sample_dir, summary_name),
            "w",
            encoding="utf-8"
        ) as f:
            f.write(str(model.summary()))
            f.write("\n\nUsed region FE: " + ", ".join(model.used_controls))
            f.write("\nDropped region FE: " + ", ".join(model.dropped_controls))

        for var in [XVAR, z_var, xz_var]:
            rows.append({
                "panel": panel_name,
                "main_variable_label": main_label,
                "z_var": z_var,
                "interaction_var": xz_var,
                "fe_spec": fe_label,
                "estimator": estimator_label,
                "weight_col": weight_col if weight_col is not None else "",
                "variable": var,
                "coef": model.params.get(var, np.nan),
                "se": model.bse.get(var, np.nan),
                "p": model.pvalues.get(var, np.nan),
                "nobs": int(model.nobs),
                "r2": model.rsquared,
                "used_region_fe": ", ".join(model.used_controls),
                "dropped_region_fe": ", ".join(model.dropped_controls),
            })

    return models, rows


def panel_cell(models, fe_label, estimator_label, var):
    key = (fe_label, estimator_label)
    if key not in models:
        return ""
    model = models[key]
    if var not in model.params.index:
        return ""
    return coef_se(model.params[var], model.bse[var], model.pvalues[var])


def panel_stat(models, fe_label, estimator_label, stat):
    key = (fe_label, estimator_label)
    if key not in models:
        return ""
    model = models[key]
    if stat == "nobs":
        return str(int(model.nobs))
    if stat == "r2":
        return f"{model.rsquared:.3f}"
    return ""


def build_panel_table(panel_models):
    """Build a single aligned panel-style table with all crop/condition panels."""
    rows = []

    
    # --------------------------------------------------------
    # BASELINE PANEL
    # --------------------------------------------------------

    baseline_models = panel_models["Baseline"]

    rows.append({
        "row": r"\multicolumn{7}{l}{\textbf{Panel A. Baseline Specification}}",
        "NoFE_OLS": "", "NoFE_WLS": "", "NoFE_WLSSQ": "",
        "FE_OLS": "", "FE_WLS": "", "FE_WLSSQ": "",
        "_panel_header": True
    })

    rows.append({
        "row": r"$\log(\hat{\bar a}_s)$",
        "NoFE_OLS": panel_cell(baseline_models, "No Region FE", "OLS", XVAR),
        "NoFE_WLS": panel_cell(baseline_models, "No Region FE", "WLS", XVAR),
        "NoFE_WLSSQ": panel_cell(baseline_models, "No Region FE", "WLS Sq.", XVAR),
        "FE_OLS": panel_cell(baseline_models, "Region FE", "OLS", XVAR),
        "FE_WLS": panel_cell(baseline_models, "Region FE", "WLS", XVAR),
        "FE_WLSSQ": panel_cell(baseline_models, "Region FE", "WLS Sq.", XVAR),
    })

    rows.append({
        "row": "Observations",
        "NoFE_OLS": panel_stat(baseline_models, "No Region FE", "OLS", "nobs"),
        "NoFE_WLS": panel_stat(baseline_models, "No Region FE", "WLS", "nobs"),
        "NoFE_WLSSQ": panel_stat(baseline_models, "No Region FE", "WLS Sq.", "nobs"),
        "FE_OLS": panel_stat(baseline_models, "Region FE", "OLS", "nobs"),
        "FE_WLS": panel_stat(baseline_models, "Region FE", "WLS", "nobs"),
        "FE_WLSSQ": panel_stat(baseline_models, "Region FE", "WLS Sq.", "nobs"),
    })

    rows.append({
        "row": r"$R^2$",
        "NoFE_OLS": panel_stat(baseline_models, "No Region FE", "OLS", "r2"),
        "NoFE_WLS": panel_stat(baseline_models, "No Region FE", "WLS", "r2"),
        "NoFE_WLSSQ": panel_stat(baseline_models, "No Region FE", "WLS Sq.", "r2"),
        "FE_OLS": panel_stat(baseline_models, "Region FE", "OLS", "r2"),
        "FE_WLS": panel_stat(baseline_models, "Region FE", "WLS", "r2"),
        "FE_WLSSQ": panel_stat(baseline_models, "Region FE", "WLS Sq.", "r2"),
    })

    for idx, (panel_title, main_label, z_var, xz_var) in enumerate(SEPARATE_INTERACTION_SPECS, start=2):

        rows.append({
            "row": rf"\multicolumn{{7}}{{l}}{{\textbf{{Panel {chr(64 + idx)}. {panel_title}}}}}",
            "NoFE_OLS": "", "NoFE_WLS": "", "NoFE_WLSSQ": "",
            "FE_OLS": "", "FE_WLS": "", "FE_WLSSQ": "",
            "_panel_header": True
        })

        models = panel_models[panel_title]

        row_specs = [
            (r"$\log(\hat{\bar a}_s)$", XVAR),
            (main_label, z_var),
            (rf"$\log(\hat{{\bar a}}_s)\times$ {main_label}", xz_var),
        ]

        for label, var in row_specs:
            rows.append({
                "row": label,
                "NoFE_OLS": panel_cell(models, "No Region FE", "OLS", var),
                "NoFE_WLS": panel_cell(models, "No Region FE", "WLS", var),
                "NoFE_WLSSQ": panel_cell(models, "No Region FE", "WLS Sq.", var),
                "FE_OLS": panel_cell(models, "Region FE", "OLS", var),
                "FE_WLS": panel_cell(models, "Region FE", "WLS", var),
                "FE_WLSSQ": panel_cell(models, "Region FE", "WLS Sq.", var),
            })

        rows.append({
            "row": "Observations",
            "NoFE_OLS": panel_stat(models, "No Region FE", "OLS", "nobs"),
            "NoFE_WLS": panel_stat(models, "No Region FE", "WLS", "nobs"),
            "NoFE_WLSSQ": panel_stat(models, "No Region FE", "WLS Sq.", "nobs"),
            "FE_OLS": panel_stat(models, "Region FE", "OLS", "nobs"),
            "FE_WLS": panel_stat(models, "Region FE", "WLS", "nobs"),
            "FE_WLSSQ": panel_stat(models, "Region FE", "WLS Sq.", "nobs"),
        })

        rows.append({
            "row": r"$R^2$",
            "NoFE_OLS": panel_stat(models, "No Region FE", "OLS", "r2"),
            "NoFE_WLS": panel_stat(models, "No Region FE", "WLS", "r2"),
            "NoFE_WLSSQ": panel_stat(models, "No Region FE", "WLS Sq.", "r2"),
            "FE_OLS": panel_stat(models, "Region FE", "OLS", "r2"),
            "FE_WLS": panel_stat(models, "Region FE", "WLS", "r2"),
            "FE_WLSSQ": panel_stat(models, "Region FE", "WLS Sq.", "r2"),
        })

    rows.append({
        "row": "Region FE",
        "NoFE_OLS": "No", "NoFE_WLS": "No", "NoFE_WLSSQ": "No",
        "FE_OLS": "Yes", "FE_WLS": "Yes", "FE_WLSSQ": "Yes",
    })

    rows.append({
        "row": "Reliability weights",
        "NoFE_OLS": "No", "NoFE_WLS": "$Match/100$", "NoFE_WLSSQ": "$(Match/100)^2$",
        "FE_OLS": "No", "FE_WLS": "$Match/100$", "FE_WLSSQ": "$(Match/100)^2$",
    })

    return pd.DataFrame(rows)


def write_panel_latex(ptable, caption, label, tex_path, sample_note):
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{3}{c}{No Region FE} & \multicolumn{3}{c}{Region FE} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append(r"& OLS & WLS & WLS Sq. & OLS & WLS & WLS Sq. \\")
    lines.append(r"\midrule")

    for _, row in ptable.iterrows():
        label_row = row["row"]
        if str(label_row).startswith(r"\multicolumn"):
            lines.append(r"\midrule")
            lines.append(label_row + r" \\")
            continue

        vals = [
            row["NoFE_OLS"], row["NoFE_WLS"], row["NoFE_WLSSQ"],
            row["FE_OLS"], row["FE_WLS"], row["FE_WLSSQ"]
        ]

        # Coef + SE cells contain line breaks; split into two table rows.
        if any("\n" in str(v) for v in vals):
            coef_vals = []
            se_vals = []
            for v in vals:
                if "\n" in str(v):
                    coef, se = str(v).split("\n", 1)
                else:
                    coef, se = str(v), ""
                coef_vals.append(coef)
                se_vals.append(se)

            lines.append(label_row + " & " + " & ".join(coef_vals) + r" \\")
            lines.append(" & " + " & ".join(se_vals) + r" \\")
        else:
            lines.append(label_row + " & " + " & ".join(map(str, vals)) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    lines.append(
        r"\item Notes: Robust HC1 standard errors are reported in parentheses. "
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
        r"The dependent variable is log SNAP participation rate, defined as log(SNAP persons divided by the July 1, 2025 Census resident population estimate). "
        + sample_note +
        r"Each panel reports a separate heterogeneous-effects specification. "
        r"Crop shares are computed using average historical cotton, corn, and soybean acreage over 1926--1940. "
        r"Suitability is average days suitable for fieldwork during April--October. "
        r"Columns 1--3 exclude region fixed effects. Columns 4--6 include region fixed effects where identified. "
        r"WLS specifications use reliability weights based on the post-calibration Soft-DTW matching score using $Match/100$. "
        r"WLS Sq. specifications use $(Match/100)^2$. "
        r"These specifications are descriptive and are not the main causal IV design."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def estimate_interactions_for_sample(base, sample_key, sample_label, sample_note, sample_dir, threshold=None):
    os.makedirs(sample_dir, exist_ok=True)

    reg, region_fe = prepare_sample(base, threshold=threshold)

    reg.to_csv(os.path.join(sample_dir, f"{sample_key}_interaction_dataset.csv"), index=False)

    print(f"\n{sample_label} states used:")
    print(sorted(reg["state"].unique()))
    print("N states:", reg["state"].nunique())
    print("Region FE candidates:", region_fe)

    all_rows = []
    panel_models = {}

    # --------------------------------------------------------
    # BASELINE ESTIMATION
    # --------------------------------------------------------

    baseline_models = {}
    baseline_rows = []

    for fe_label, estimator_label, weight_col in TABLE_COLS:

        controls = [] if fe_label == "No Region FE" else region_fe

        try:
            model, used_data = run_ols_or_wls(
                y=YVAR,
                xvars=[XVAR],
                controls=controls,
                data=reg,
                weight_col=weight_col
            )
        except Exception as e:
            print(f"Skipped baseline | {fe_label} | {estimator_label}: {repr(e)}")
            continue

        baseline_models[(fe_label, estimator_label)] = model

        baseline_rows.append({
            "panel": "Baseline",
            "main_variable_label": "",
            "z_var": "",
            "interaction_var": "",
            "fe_spec": fe_label,
            "estimator": estimator_label,
            "weight_col": weight_col if weight_col is not None else "",
            "variable": XVAR,
            "coef": model.params.get(XVAR, np.nan),
            "se": model.bse.get(XVAR, np.nan),
            "p": model.pvalues.get(XVAR, np.nan),
            "nobs": int(model.nobs),
            "r2": model.rsquared,
            "used_region_fe": ", ".join(model.used_controls),
            "dropped_region_fe": ", ".join(model.dropped_controls),
        })

    panel_models["Baseline"] = baseline_models
    all_rows.extend(baseline_rows)

    # --------------------------------------------------------
    # INTERACTION ESTIMATION
    # --------------------------------------------------------

    for panel_title, main_label, z_var, xz_var in SEPARATE_INTERACTION_SPECS:
        models, rows = estimate_one_panel(
            reg=reg,
            region_fe=region_fe,
            panel_name=panel_title,
            main_label=main_label,
            z_var=z_var,
            xz_var=xz_var,
            sample_key=sample_key,
            sample_dir=sample_dir
        )
        panel_models[panel_title] = models
        all_rows.extend(rows)

    results_table = pd.DataFrame(all_rows)
    results_table["sample"] = sample_label
    results_table["sample_key"] = sample_key
    results_table["threshold"] = threshold

    results_table.to_csv(os.path.join(sample_dir, f"{sample_key}_interaction_results_long.csv"), index=False)

    ptable = build_panel_table(panel_models)
    ptable.to_csv(os.path.join(sample_dir, f"{sample_key}_panel_publishable_table.csv"), index=False)

    tex_path = os.path.join(sample_dir, f"{sample_key}_panel_interaction_table.tex")
    write_panel_latex(
        ptable=ptable,
        caption=f"{sample_label} Heterogeneous-Effects Specifications: SNAP Participation Rate",
        label=f"tab:{sample_key}_heterogeneous_effects",
        tex_path=tex_path,
        sample_note=sample_note,
    )

    print("\n==============================")
    print(f"{sample_label.upper()} PANEL INTERACTION TABLE")
    print("==============================")
    print(ptable)

    return {
        "sample_key": sample_key,
        "sample_label": sample_label,
        "sample_dir": sample_dir,
        "threshold": threshold,
        "reg": reg,
        "results_table": results_table,
        "ptable": ptable,
        "tex_path": tex_path,
    }


def make_threshold_summary_table(all_threshold_outputs, out_dir):
    """Compact appendix table for separate interaction robustness.

    Reports only the interaction coefficients for each crop/condition panel,
    across thresholds and the six specifications:
      No FE OLS, No FE WLS, No FE WLS Sq.,
      FE OLS, FE WLS, FE WLS Sq.
    """

    rows = []

    for output in all_threshold_outputs:
        res = output["results_table"].copy()
        threshold = output["threshold"]
        sample = f"Match score $>{int(threshold)}\\%$"

        for fe_label, estimator_label, _ in TABLE_COLS:
            spec_label = f"{'No FE' if fe_label == 'No Region FE' else 'FE'} {estimator_label}"
            row = {
                "Sample": sample,
                "Specification": spec_label,
            }

            nobs_vals = []
            r2_vals = []

            for panel_title, main_label, z_var, xz_var in SEPARATE_INTERACTION_SPECS:
                sub = res[
                    (res["panel"] == panel_title) &
                    (res["fe_spec"] == fe_label) &
                    (res["estimator"] == estimator_label) &
                    (res["variable"] == xz_var)
                ].copy()

                if sub.empty:
                    row[rf"$\log(\hat{{\bar a}}_s)\times$ {main_label}"] = ""
                else:
                    one = sub.iloc[0]
                    row[rf"$\log(\hat{{\bar a}}_s)\times$ {main_label}"] = coef_se(
                        one["coef"], one["se"], one["p"]
                    )
                    nobs_vals.append(one["nobs"])
                    r2_vals.append(one["r2"])

            row["Observations"] = int(nobs_vals[0]) if nobs_vals else ""
            row["Mean $R^2$"] = f"{np.mean(r2_vals):.3f}" if r2_vals else ""
            rows.append(row)

    summary = pd.DataFrame(rows)

    col_order = [
        "Sample",
        "Specification",
        r"$\log(\hat{\bar a}_s)\times$ Cotton share",
        r"$\log(\hat{\bar a}_s)\times$ Corn share",
        r"$\log(\hat{\bar a}_s)\times$ Soybean share",
        r"$\log(\hat{\bar a}_s)\times$ Suitability",
        "Observations",
        "Mean $R^2$",
    ]

    for c in col_order:
        if c not in summary.columns:
            summary[c] = ""

    summary = summary[col_order]
    summary.to_csv(os.path.join(out_dir, "interaction_threshold_summary.csv"), index=False)

    latex = summary.to_latex(index=False, escape=False, column_format="llcccccc")

    latex_full = (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        "\\caption{Separate Interaction Robustness Across Calibration-Quality Thresholds}\n"
        "\\label{tab:separate_interaction_threshold_robustness}\n"
        "\\begin{threeparttable}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex +
        "}\n"
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        "\\item Notes: Robust HC1 standard errors are reported in parentheses. "
        "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
        "The dependent variable is log SNAP participation rate, defined as log(SNAP persons divided by the July 1, 2025 Census resident population estimate). "
        "Each entry reports the interaction coefficient from a separate heterogeneous-effects specification. "
        "No-FE rows exclude region fixed effects; FE rows include region fixed effects where identified. "
        "Reliability weights use the post-calibration Soft-DTW matching score. "
        "These estimates are descriptive robustness checks and are not the main causal IV design.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}\n"
        "\\end{table}\n"
    )

    tex_path = os.path.join(out_dir, "interaction_threshold_summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_full)

    return summary, tex_path



# ============================================================
# MAIN RUN: SELECTED SAMPLE INTERACTION TABLE ONLY
# ============================================================
# This script estimates the published selected-sample interaction table and
# copies only that manuscript-facing table to outputs/regression/interaction_table.tex.

selected_gt90_dir = os.path.join(output_dir, "gt90_main")

selected_gt90_output = estimate_interactions_for_sample(
    base=base_reg,
    sample_key="gt90_int",
    sample_label="Selected-State, Match Score $>$ 90\\%",
    sample_note="The selected sample includes states with post-calibration Soft-DTW matching scores greater than 90 percent. ",
    sample_dir=selected_gt90_dir,
    threshold=MAIN_SELECTED_MATCH_THRESHOLD,
)

# Copy only the manuscript-facing interaction table.
interaction_source = selected_gt90_output["tex_path"]
interaction_target = MANUSCRIPT_REGRESSION_DIR / "interaction_table.tex"
if os.path.exists(interaction_source):
    shutil.copyfile(interaction_source, interaction_target)
else:
    raise FileNotFoundError(f"Expected interaction table not found: {interaction_source}")

print("\nFinal manuscript-facing interaction output:")
print(interaction_target)
