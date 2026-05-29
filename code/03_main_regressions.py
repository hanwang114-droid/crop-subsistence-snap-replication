# -----------------------------------------------------------------------------
# This script is part of the replication workflow. It assumes that
# code/00_clean_data.py has already converted original files in raw_data/ into
# clean_data/. This script reads cleaned inputs only; it does not read raw_data/.
# -----------------------------------------------------------------------------

"""Estimate the main SNAP regressions and IV robustness table.

Inputs:
  * baseline calibrated subsistence index from 02_calibration_model.py;
  * SNAP participation rate = the latest SNAP persons column / July 1, 2025 population;
  * historical crop acreage per state land area averaged over 1926--1940;
  * fieldwork suitability data.

Published outputs:
  * main_regression_table.tex: baseline OLS and IV estimates;
  * first_stage_table.tex: first-stage estimates for the IV specifications;
  * soybean_iv_threshold_robustness_compact_table.tex: soybean IV robustness
    across matching-score thresholds.

The crop instruments use log(1 + crop acres per 1,000 state land acres), which
keeps zero-crop states in the sample, reduces leverage from very large values,
and adjusts historical crop exposure for state land area.
"""
# -----------------------------------------------------------------------------
# READER GUIDE
# -----------------------------------------------------------------------------
# This file estimates the paper's main state-level SNAP regressions. It reads
# the calibrated index from 02_calibration_model.py, constructs SNAP rates,
# historical log(1 + acreage-intensity) instruments, and reliability weights. It writes
# only publication-facing regression tables to outputs/regression/.
# -----------------------------------------------------------------------------

# ============================================================
# MATCHING-SCORE-SELECTED AND FULL-SAMPLE IV REGRESSIONS
# LOG SUBSISTENCE PARAMETER + LOG HISTORICAL CROP ACREAGE-INTENSITY IVs
# ============================================================
#
# This script uses the optimized-alpha Stone-Geary calibration output:
#   calibrated_state_abar_optimized_alpha_raw_productivity.csv
#
# Main empirical specification:
#   Outcome:      log SNAP participation rate = log(SNAP persons / 2025 resident population)
#   Endogenous:   log(abar_state)
#   Instruments:  log(1 + historical crop acres per 1,000 state land acres),
#                 averaged over 1926--1940
#                 plus fieldwork suitability as an alternative IV
#
# Selected-state sample is dynamic:
#   matching_score_percent > 95
#
# The old crop-share IVs are not used in the main IV specifications here.
# The crop-share construction was mechanically relative; log acreage gives a
# more direct measure of historical crop exposure after adjusting for state size.
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
try:
    from linearmodels.iv import IV2SLS
    HAVE_LINEARMODELS = True
except Exception:
    from statsmodels.sandbox.regression.gmm import IV2SLS as SMIV2SLS
    HAVE_LINEARMODELS = False
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# PATHS
# ============================================================

crop_files = {
    "soybean": str(ROOT / "clean_data" / "cash_crop_data" / "soybean.csv"),
    "corn": str(ROOT / "clean_data" / "cash_crop_data" / "corn.csv"),
    "cotton": str(ROOT / "clean_data" / "cash_crop_data" / "cotton.csv"),
}

suitability_path = str(ROOT / "clean_data" / "cash_crop_data" / "suitability_clean.csv")
state_land_area_path = str(ROOT / "clean_data" / "census_gazetteer" / "state_land_area_2024.csv")
abar_path = str(ROOT / "derived" / "calibration" / "calibrated_state_abar_optimized_alpha_raw_productivity.csv")
quality_path = str(ROOT / "derived" / "calibration" / "soft_dtw_matching_quality_by_state_raw_productivity.csv")
snap_path = str(ROOT / "clean_data" / "snap" / "snap-persons-4.xlsx")
population_path = str(ROOT / "clean_data" / "snap" / "NST-EST2025-POP.xlsx")

# Internal work directory for detailed regression outputs used to build the
# publication tables. The final manuscript-facing tables are copied to
# outputs/regression/ at the end of the script.
output_dir = str(ROOT / "derived" / "regression_work")
os.makedirs(output_dir, exist_ok=True)
MANUSCRIPT_REGRESSION_DIR = ROOT / "outputs" / "regression"
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
YVAR_PERSONS = "log_snap_persons"
SNAP_RATE = "snap_rate"
POP_YEAR = 2025
ABAR_RAW = "abar_state"
XVAR = "log_abar_state"
MATCH = "matching_score_percent"

SELECTED_MATCH_THRESHOLD = 95.0

CROPS = ["cotton", "corn", "soybean"]
CROP_START_YEAR = 1926
CROP_END_YEAR = 1940

SUIT = "suitability"
START_MONTH = 4
END_MONTH = 10

# Use log1p for crop acreage intensity so that zero-crop states remain in the sample.
# log1p(x) = log(1 + x), where x is crop acres per 1,000 state land acres.
USE_LOG1P_CROP_ACRES = True

WEIGHT_SPECS = [
    ("Unweighted", None, "No"),
    ("WLS", "match_weight", "$Match/100$"),
    ("WLS Sq.", "match_weight_sq", "$(Match/100)^2$"),
]

IV_SPECS = [
    ("Log Cotton Acreage Intensity IV", "log_cotton_acres", "Log cotton acreage intensity"),
    ("Log Corn Acreage Intensity IV", "log_corn_acres", "Log corn acreage intensity"),
    ("Log Soybean Acreage Intensity IV", "log_soybean_acres", "Log soybean acreage intensity"),
    ("Suitability IV", SUIT, "Suitability"),
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
    return name[:180]


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
    return f"{coef:.3f}{stars(p)}\n({se:.3f})"


def clean_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def load_census_state_population(population_path, state_to_abbr, pop_year=2025):
    """Load Census NST-EST2025-POP.xlsx and return state-level resident population.

    The Census workbook has a multi-row header. This function is deliberately
    defensive: it reads the workbook without a fixed header, finds the
    "Geographic Area" row, identifies the requested year column, and keeps only
    state rows whose names map to state abbreviations.
    """
    raw = pd.read_excel(population_path, sheet_name=0, header=None)

    geo_row_candidates = raw.index[
        raw.apply(lambda r: r.astype(str).str.contains("Geographic Area", case=False, na=False).any(), axis=1)
    ].tolist()
    if not geo_row_candidates:
        raise ValueError("Could not find the 'Geographic Area' header row in the Census population workbook.")

    header_row = geo_row_candidates[0]
    year_row = header_row + 1

    # The year labels are usually on the row immediately below the main header.
    year_values = pd.to_numeric(raw.loc[year_row], errors="coerce")
    year_cols = [col for col, val in year_values.items() if pd.notna(val) and int(val) == int(pop_year)]
    if not year_cols:
        raise ValueError(f"Could not find a Census population column for year {pop_year}.")
    pop_col = year_cols[-1]

    geo_col = raw.loc[header_row].astype(str).str.contains("Geographic Area", case=False, na=False)
    geo_cols = list(raw.columns[geo_col])
    if not geo_cols:
        raise ValueError("Could not identify the Census geographic-area column.")
    geo_col = geo_cols[0]

    pop = raw.loc[year_row + 1:, [geo_col, pop_col]].copy()
    pop.columns = ["state_name_raw", f"population_{pop_year}"]

    pop["state_name"] = (
        pop["state_name_raw"]
        .astype(str)
        .str.replace(r"^\.", "", regex=True)
        .str.strip()
        .str.title()
    )
    pop["state"] = pop["state_name"].map(state_to_abbr)
    pop[f"population_{pop_year}"] = pd.to_numeric(pop[f"population_{pop_year}"], errors="coerce")

    pop = pop.dropna(subset=["state", f"population_{pop_year}"]).copy()
    pop = pop[pop[f"population_{pop_year}"] > 0].copy()
    pop = pop[["state", "state_name", f"population_{pop_year}"]].drop_duplicates("state")

    if pop["state"].nunique() < 48:
        raise ValueError(
            f"Population merge found only {pop['state'].nunique()} states. "
            "Check the Census workbook layout or state-name mapping."
        )

    return pop


def filter_weights(d, weight_col):
    if weight_col is None:
        return d
    return d[np.isfinite(d[weight_col]) & (d[weight_col] > 0)].copy()


def full_rank_controls(d, controls, base_cols=None):
    base_cols = [] if base_cols is None else list(base_cols)
    kept = []

    X_parts = [np.ones((len(d), 1))]
    if base_cols:
        X_parts.append(d[base_cols].to_numpy(dtype=float))

    X = np.column_stack(X_parts)
    current_rank = np.linalg.matrix_rank(X)

    for col in controls:
        if col not in d.columns:
            continue
        trial = np.column_stack([X, d[[col]].to_numpy(dtype=float)])
        trial_rank = np.linalg.matrix_rank(trial)
        if trial_rank > current_rank:
            kept.append(col)
            X = trial
            current_rank = trial_rank

    dropped = [c for c in controls if c not in kept]
    return kept, dropped


def run_ols(y, xvars, controls, data, weight_col=None):
    needed = [y] + xvars + controls
    if weight_col is not None:
        needed.append(weight_col)

    d = data.dropna(subset=needed).copy()
    d = clean_numeric(d, needed).dropna(subset=needed)
    d = filter_weights(d, weight_col)

    if len(d) < 4:
        raise ValueError(f"Too few observations for OLS/WLS: {len(d)}")

    controls_kept, controls_dropped = full_rank_controls(d, controls, base_cols=xvars)
    used_xvars = xvars + controls_kept
    X = sm.add_constant(d[used_xvars], has_constant="add")

    if np.linalg.matrix_rank(X.to_numpy(dtype=float)) < X.shape[1]:
        raise ValueError("OLS regressors are not full rank.")

    if weight_col is None:
        model = sm.OLS(d[y], X).fit(cov_type="HC1")
    else:
        model = sm.WLS(d[y], X, weights=d[weight_col]).fit(cov_type="HC1")

    model.used_controls = controls_kept
    model.dropped_controls = controls_dropped
    return model, d


def run_first_stage(x, z, controls, data, weight_col=None):
    needed = [x, z] + controls
    if weight_col is not None:
        needed.append(weight_col)

    d = data.dropna(subset=needed).copy()
    d = clean_numeric(d, needed).dropna(subset=needed)
    d = filter_weights(d, weight_col)

    controls_kept, controls_dropped = full_rank_controls(d, controls, base_cols=[z])
    X = sm.add_constant(d[[z] + controls_kept], has_constant="add")

    if np.linalg.matrix_rank(X.to_numpy(dtype=float)) < X.shape[1]:
        raise ValueError("First-stage regressors are not full rank.")

    if weight_col is None:
        model = sm.OLS(d[x], X).fit(cov_type="HC1")
    else:
        model = sm.WLS(d[x], X, weights=d[weight_col]).fit(cov_type="HC1")

    ftest = model.f_test(f"{z} = 0")
    fstat = float(np.asarray(ftest.fvalue).item())

    model.used_controls = controls_kept
    model.dropped_controls = controls_dropped
    return model, fstat, controls_kept, controls_dropped


def run_iv(y, x, z, controls, data, weight_col=None):
    needed = [y, x, z] + controls
    if weight_col is not None:
        needed.append(weight_col)

    d = data.dropna(subset=needed).copy()
    d = clean_numeric(d, needed).dropna(subset=needed)
    d = filter_weights(d, weight_col)

    if len(d) < 8:
        raise ValueError(f"Too few observations for IV: {len(d)}")

    controls_kept, controls_dropped = full_rank_controls(d, controls, base_cols=None)

    exog = pd.DataFrame({"const": 1.0}, index=d.index)
    if controls_kept:
        exog = pd.concat([exog, d[controls_kept]], axis=1)

    endog = d[[x]]
    instr = d[[z]]

    rank_x = np.linalg.matrix_rank(pd.concat([exog, endog], axis=1))
    rank_z = np.linalg.matrix_rank(pd.concat([exog, instr], axis=1))
    if rank_x < exog.shape[1] + 1:
        raise ValueError("IV regressors [exog endog] are not full rank.")
    if rank_z < exog.shape[1] + 1:
        raise ValueError("IV instruments [exog instruments] are not full rank.")

    if HAVE_LINEARMODELS:
        if weight_col is None:
            model = IV2SLS(d[y], exog, endog, instr).fit(cov_type="robust")
        else:
            model = IV2SLS(d[y], exog, endog, instr, weights=d[weight_col]).fit(cov_type="robust")
    else:
        # Portable fallback using statsmodels' sandbox IV2SLS. When weights are
        # requested, variables are premultiplied by sqrt(weight). This preserves
        # the same weighted moment structure, although standard errors may not be
        # numerically identical to linearmodels' robust covariance estimates.
        X_all = pd.concat([exog, endog], axis=1)
        Z_all = pd.concat([exog, instr], axis=1)
        yy = d[y].astype(float)
        if weight_col is not None:
            sw = np.sqrt(d[weight_col].astype(float))
            yy = yy * sw
            X_all = X_all.mul(sw, axis=0)
            Z_all = Z_all.mul(sw, axis=0)
        y_arr = yy.to_numpy(dtype=float).reshape(-1, 1)
        X_arr = X_all.to_numpy(dtype=float)
        Z_arr = Z_all.to_numpy(dtype=float)
        sm_res = SMIV2SLS(y_arr.ravel(), X_arr, Z_arr).fit()

        beta = np.asarray(sm_res.params, dtype=float).reshape(-1, 1)
        resid = y_arr - X_arr @ beta
        ztz_inv = np.linalg.pinv(Z_arr.T @ Z_arr)
        bread_inv = X_arr.T @ Z_arr @ ztz_inv @ Z_arr.T @ X_arr
        bread = np.linalg.pinv(bread_inv)
        meat_middle = Z_arr.T @ ((resid ** 2) * Z_arr)
        meat = X_arr.T @ Z_arr @ ztz_inv @ meat_middle @ ztz_inv @ Z_arr.T @ X_arr
        cov = bread @ meat @ bread
        nobs = X_arr.shape[0]
        k = X_arr.shape[1]
        if nobs > k:
            cov *= nobs / (nobs - k)  # HC1 small-sample correction
        se = np.sqrt(np.diag(cov))
        tvals = beta.ravel() / se
        from scipy import stats
        pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals)))

        class IVFallbackResult:
            pass

        model = IVFallbackResult()
        model.params = pd.Series(beta.ravel(), index=X_all.columns)
        model.bse = pd.Series(se, index=X_all.columns)
        model.pvalues = pd.Series(pvals, index=X_all.columns)
        model.nobs = int(nobs)
        y_orig = d[y].to_numpy(dtype=float)
        fitted_orig = (pd.concat([exog, endog], axis=1).to_numpy(dtype=float) @ beta).ravel()
        sse = float(np.sum((y_orig - fitted_orig) ** 2))
        sst = float(np.sum((y_orig - y_orig.mean()) ** 2))
        model.rsquared = 1 - sse / sst if sst > 0 else np.nan
        model.summary = sm_res.summary()

    # First-stage is computed on the exact IV estimation sample, with the exact same controls/weights.
    fs, fstat, _, _ = run_first_stage(x=x, z=z, controls=controls_kept, data=d, weight_col=weight_col)

    model.used_controls = controls_kept
    model.dropped_controls = controls_dropped
    model.first_stage_F = fstat
    model.first_stage_coef = fs.params.get(z, np.nan)
    model.first_stage_se = fs.bse.get(z, np.nan)
    model.first_stage_p = fs.pvalues.get(z, np.nan)
    model.first_stage_r2 = fs.rsquared
    return model, fs, d


def instrument_display_name(instrument):
    labels = {
        "None": "None",
        "log_cotton_acres": "Log cotton acreage intensity",
        "log_corn_acres": "Log corn acreage intensity",
        "log_soybean_acres": "Log soybean acreage intensity",
        SUIT: "Suitability",
    }
    return labels.get(instrument, instrument)


def make_result_row(model_name, weight_label, weight_display, fe_label, instrument,
                    model, coef_name, first_stage_F=np.nan, first_stage_coef=np.nan,
                    first_stage_se=np.nan, first_stage_p=np.nan, first_stage_r2=np.nan):
    if hasattr(model, "std_errors"):
        se = model.std_errors[coef_name]
        pval = model.pvalues[coef_name]
        r2 = model.rsquared
    else:
        se = model.bse[coef_name]
        pval = model.pvalues[coef_name]
        r2 = model.rsquared

    return {
        "model": model_name,
        "weighting": weight_label,
        "weight_display": weight_display,
        "fe_spec": fe_label,
        "instrument": instrument,
        "coef": model.params[coef_name],
        "se": se,
        "p": pval,
        "nobs": int(model.nobs),
        "r2": r2,
        "first_stage_coef": first_stage_coef,
        "first_stage_se": first_stage_se,
        "first_stage_p": first_stage_p,
        "first_stage_F": first_stage_F,
        "first_stage_r2": first_stage_r2,
        "region_fe": "Yes" if fe_label == "Region FE" else "No",
        "used_controls": ", ".join(model.used_controls),
        "dropped_controls": ", ".join(model.dropped_controls),
    }


def build_single_model_table(results_table, model_name):
    ordered_cols = []
    colnames = []

    for fe_label in ["No region FE", "Region FE"]:
        for weight_label, _, _ in WEIGHT_SPECS:
            ordered_cols.append((fe_label, weight_label))
            if fe_label == "No region FE":
                colnames.append(weight_label)
            else:
                colnames.append(f"{weight_label} + Region FE")

    row_index = [
        "$\\log(\\hat{\\bar a}_s)$",
        "",
        "First-stage coefficient",
        "",
        "Region FE",
        "Reliability weights",
        "Instrument",
        "First-stage F-stat",
        "Observations",
        "$R^2$",
    ]

    table = pd.DataFrame(index=row_index, columns=colnames)

    for col, (fe_label, weight_label) in zip(colnames, ordered_cols):
        row = results_table[
            (results_table["model"] == model_name) &
            (results_table["fe_spec"] == fe_label) &
            (results_table["weighting"] == weight_label)
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        table.loc["$\\log(\\hat{\\bar a}_s)$", col] = coef_se(row["coef"], row["se"], row["p"])

        if row["instrument"] != "None" and pd.notna(row["first_stage_coef"]):
            table.loc["First-stage coefficient", col] = coef_se(
                row["first_stage_coef"], row["first_stage_se"], row["first_stage_p"]
            )

        table.loc["Region FE", col] = row["region_fe"]
        table.loc["Reliability weights", col] = row["weight_display"]
        table.loc["Instrument", col] = instrument_display_name(row["instrument"])
        if pd.notna(row["first_stage_F"]):
            table.loc["First-stage F-stat", col] = f"{row['first_stage_F']:.2f}"
        table.loc["Observations", col] = int(row["nobs"])
        table.loc["$R^2$", col] = f"{row['r2']:.3f}"

    return table.fillna("")


def write_latex_table_for_output(table, caption, label, filename, output_dir_for_run, sample_note):
    latex = table.to_latex(escape=False, column_format="l" + "c" * table.shape[1], multicolumn=False)
    latex = latex.replace(
        "\\toprule",
        "\\toprule\n"
        "& \\multicolumn{3}{c}{No Region FE} "
        "& \\multicolumn{3}{c}{Region FE} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"
    )

    latex_full = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex +
        "}\n"
        "\\begin{flushleft}\n"
        "\\footnotesize\n"
        "Notes: Robust standard errors in parentheses. "
        "* $p<0.10$, ** $p<0.05$, *** $p<0.01$. "
        "Dependent variable is log SNAP participation rate, defined as log(SNAP persons divided by 2025 resident population). "
        "The endogenous regressor in IV specifications is $\\log(\\hat{\\bar a}_s)$. "
        "First-stage coefficients report the coefficient on the excluded instrument in regressions of "
        "$\\log(\\hat{\\bar a}_s)$ on the instrument and included controls. "
        f"{sample_note} "
        "Crop instruments are log(1 + historical crop acres per 1,000 state land acres) averaged over 1926--1940. "
        "Suitability is average days suitable for fieldwork during April--October. "
        "Reliability weights use the post-calibration Soft-DTW matching score. "
        "\\end{flushleft}\n"
        "\\end{table}\n"
    )

    outpath = os.path.join(output_dir_for_run, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(latex_full)
    return outpath

assert_baseline_calibration_method()

# ============================================================
# LOAD AND CLEAN CROP DATA: HISTORICAL ACREAGE
# ============================================================

cleaned = []
for crop_name, path in crop_files.items():
    temp = pd.read_csv(path)
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
    cleaned.append(temp)

crop_panel = pd.concat(cleaned, ignore_index=True)
crop_panel = crop_panel.dropna(subset=["state", "year", "value"])
crop_panel["year"] = crop_panel["year"].astype(int)
crop_panel = crop_panel[(crop_panel["year"] >= CROP_START_YEAR) & (crop_panel["year"] <= CROP_END_YEAR)].copy()

crop_state = crop_panel.groupby(["state", "crop"], as_index=False)["value"].mean()
crop_wide = crop_state.pivot(index="state", columns="crop", values="value").reset_index()

for crop in CROPS:
    if crop not in crop_wide.columns:
        crop_wide[crop] = 0.0

crop_wide[CROPS] = crop_wide[CROPS].fillna(0.0)

state_land = pd.read_csv(state_land_area_path)
state_land["state"] = state_land["state"].astype(str).str.upper().str.strip()
state_land["land_acres"] = pd.to_numeric(state_land["land_acres"], errors="coerce")
crop_wide = crop_wide.merge(state_land[["state", "land_acres"]], on="state", how="left")
if crop_wide["land_acres"].isna().any():
    missing_land = sorted(crop_wide.loc[crop_wide["land_acres"].isna(), "state"].dropna().unique())
    raise ValueError(f"Missing state land area for crop states: {missing_land}")

for crop in CROPS:
    crop_wide[f"{crop}_acres_1926_1940"] = crop_wide[crop]
    crop_wide[f"{crop}_acres_per_1000_land_acres"] = np.where(
        crop_wide["land_acres"] > 0,
        1000 * crop_wide[crop] / crop_wide["land_acres"],
        np.nan
    )
    if USE_LOG1P_CROP_ACRES:
        crop_wide[f"log_{crop}_acres"] = np.log1p(crop_wide[f"{crop}_acres_per_1000_land_acres"])
    else:
        crop_wide[f"log_{crop}_acres"] = np.where(
            crop_wide[f"{crop}_acres_per_1000_land_acres"] > 0,
            np.log(crop_wide[f"{crop}_acres_per_1000_land_acres"]),
            np.nan
        )

# Keep crop shares as diagnostics/robustness variables, but not as main IVs.
crop_wide["cash_crop_sum"] = crop_wide[CROPS].sum(axis=1)
for crop in CROPS:
    crop_wide[f"share_{crop}"] = np.where(crop_wide["cash_crop_sum"] > 0, crop_wide[crop] / crop_wide["cash_crop_sum"], np.nan)

# ============================================================
# LOAD SUITABILITY
# ============================================================

suit = pd.read_csv(suitability_path)
suit["state_name"] = suit["state"].astype(str).str.title().str.strip()
suit["state"] = suit["state_name"].map(state_to_abbr)
suit["week_ending"] = pd.to_datetime(suit["week_ending"], errors="coerce")
suit["month"] = suit["week_ending"].dt.month
suit[SUIT] = pd.to_numeric(suit[SUIT], errors="coerce")
suit = suit[suit["month"].between(START_MONTH, END_MONTH)].copy()
suit = suit.groupby(["state", "state_name"], as_index=False)[SUIT].mean()

# ============================================================
# LOAD ABAR + MATCHING QUALITY
# ============================================================

abar = pd.read_csv(abar_path)
quality = pd.read_csv(quality_path)

abar["state"] = abar["state"].astype(str).str.upper().str.strip()
quality["state"] = quality["state"].astype(str).str.upper().str.strip()

df_abar = abar.merge(quality[["state", MATCH]], on="state", how="left")
df_abar = clean_numeric(df_abar, [ABAR_RAW, MATCH])

# Log subsistence parameter: requires positive calibrated abar values.
df_abar = df_abar[df_abar[ABAR_RAW] > 0].copy()
df_abar[XVAR] = np.log(df_abar[ABAR_RAW])

df_abar["match_weight"] = df_abar[MATCH] / 100
df_abar["match_weight_sq"] = df_abar["match_weight"] ** 2
df_abar.loc[df_abar["match_weight"] <= 0, "match_weight"] = np.nan
df_abar.loc[df_abar["match_weight_sq"] <= 0, "match_weight_sq"] = np.nan

# ============================================================
# LOAD SNAP + CENSUS POPULATION
# ============================================================

snap = pd.read_excel(snap_path, sheet_name=0, header=2)
snap = snap.rename(columns={snap.columns[0]: "state_name"})
snap["state_name"] = snap["state_name"].astype(str).str.title().str.strip()
snap["state"] = snap["state_name"].map(state_to_abbr)

# The SNAP workbook usually contains several date columns. This keeps your
# existing rule: use the last usable numeric column among the first few data
# columns. The selected column is printed and saved in the regression dataset.
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
snap[YVAR_PERSONS] = np.log(snap["snap_persons"])

# Census resident population estimates. For the latest SNAP month, the closest
# official complete annual population denominator is July 1, 2025.
population = load_census_state_population(
    population_path=population_path,
    state_to_abbr=state_to_abbr,
    pop_year=POP_YEAR
)

snap = snap.merge(
    population[["state", f"population_{POP_YEAR}"]],
    on="state",
    how="inner"
)

snap[SNAP_RATE] = snap["snap_persons"] / snap[f"population_{POP_YEAR}"]
snap = snap[snap[SNAP_RATE] > 0].copy()
snap[YVAR] = np.log(snap[SNAP_RATE])

snap = snap[[
    "state",
    "state_name",
    "snap_persons",
    f"population_{POP_YEAR}",
    SNAP_RATE,
    YVAR_PERSONS,
    YVAR,
]].copy()

print("SNAP column used:", snap_used_column)
print(f"Population denominator: Census resident population, July 1, {POP_YEAR}")
print("SNAP-rate states after merge:", snap["state"].nunique())


# ============================================================
# PANEL-STYLE LATEX TABLES: FIRST STAGE AND BASELINE/IV ESTIMATES
# ============================================================

PANEL_COLS = [
    ("No region FE", "Unweighted", "OLS"),
    ("No region FE", "WLS", "WLS"),
    ("No region FE", "WLS Sq.", "WLS Sq."),
    ("Region FE", "Unweighted", "OLS"),
    ("Region FE", "WLS", "WLS"),
    ("Region FE", "WLS Sq.", "WLS Sq."),
]

MODEL_PANEL_ORDER = [
    ("Baseline", "Panel A. Baseline OLS"),
    ("Log Cotton Acreage Intensity IV", "Panel B. Log Cotton Acreage Intensity IV"),
    ("Log Corn Acreage Intensity IV", "Panel C. Log Corn Acreage Intensity IV"),
    ("Log Soybean Acreage Intensity IV", "Panel D. Log Soybean Acreage Intensity IV"),
    ("Suitability IV", "Panel E. Suitability IV"),
]

FIRST_STAGE_PANEL_ORDER = [
    ("Log Cotton Acreage Intensity IV", "Panel A. Log Cotton Acreage Intensity First Stage"),
    ("Log Corn Acreage Intensity IV", "Panel B. Log Corn Acreage Intensity First Stage"),
    ("Log Soybean Acreage Intensity IV", "Panel C. Log Soybean Acreage Intensity First Stage"),
    ("Suitability IV", "Panel D. Suitability First Stage"),
]


def latex_coef_se(coef, se, p):
    if pd.isna(coef) or pd.isna(se):
        return ""
    st = stars(p)
    if st:
        return f"${coef:.3f}^{{{st}}}$\n$({se:.3f})$"
    return f"${coef:.3f}$\n$({se:.3f})$"


def _get_result(results_table, model_name, fe_label, weight_label):
    row = results_table[
        (results_table["model"] == model_name) &
        (results_table["fe_spec"] == fe_label) &
        (results_table["weighting"] == weight_label)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def _split_coef_cell(cell):
    if isinstance(cell, str) and "\n" in cell:
        return cell.split("\n", 1)
    return cell, ""


def _write_panel_table_from_rows(rows, caption, label, outpath, notes):
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\hline\hline")
    lines.append(r"& \multicolumn{3}{c}{No Region FE} & \multicolumn{3}{c}{Region FE} \\")
    lines.append(r"& OLS & WLS & WLS Sq. & OLS & WLS & WLS Sq. \\")
    lines.append(r"\hline")

    seen_panel = False
    for row in rows:
        if row.get("panel"):
            if seen_panel:
                lines.append(r"\hline")
            lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{row['panel']}}}}} \\")
            seen_panel = True
            continue

        vals = row["vals"]
        if any(isinstance(v, str) and "\n" in v for v in vals):
            coefs, ses = [], []
            for v in vals:
                c, s = _split_coef_cell(v)
                coefs.append(c)
                ses.append(s)
            lines.append(row["label"] + " & " + " & ".join(coefs) + r" \\")
            lines.append(" & " + " & ".join(ses) + r" \\")
        else:
            lines.append(row["label"] + " & " + " & ".join(str(v) for v in vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"Region FE & No & No & No & Yes & Yes & Yes \\")
    lines.append(r"Reliability weights & No & $Match/100$ & $(Match/100)^2$ & No & $Match/100$ & $(Match/100)^2$ \\")
    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\vspace{0.25em}")
    lines.append(r"\begin{minipage}{0.98\textwidth}")
    lines.append(r"\footnotesize")
    note_text = notes.replace("Notes:", r"\emph{Notes:}", 1)
    lines.append(note_text)
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return outpath


def build_baseline_iv_panel_rows(results_table):
    rows = []

    for model_name, panel_title in MODEL_PANEL_ORDER:
        rows.append({"panel": panel_title})

        # Second-stage / OLS coefficient on log(abar)
        coef_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            if r is None:
                coef_vals.append("")
            else:
                coef_vals.append(latex_coef_se(r["coef"], r["se"], r["p"]))
        rows.append({"label": r"$\log(\hat{\bar a}_s)$", "vals": coef_vals})

        # For IV panels, report first-stage F-stat as a compact diagnostic.
        if model_name != "Baseline":
            f_vals = []
            for fe_label, weight_label, _ in PANEL_COLS:
                r = _get_result(results_table, model_name, fe_label, weight_label)
                if r is None or pd.isna(r["first_stage_F"]):
                    f_vals.append("")
                else:
                    f_vals.append(f"{r['first_stage_F']:.2f}")
            rows.append({"label": "First-stage F-stat", "vals": f_vals})

        # Observations
        n_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            n_vals.append("" if r is None else str(int(r["nobs"])))
        rows.append({"label": "Observations", "vals": n_vals})

        # R2
        r2_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            r2_vals.append("" if r is None else f"{r['r2']:.3f}")
        rows.append({"label": r"$R^2$", "vals": r2_vals})

    return rows


def build_first_stage_panel_rows(results_table):
    rows = []

    for model_name, panel_title in FIRST_STAGE_PANEL_ORDER:
        rows.append({"panel": panel_title})

        coef_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            if r is None:
                coef_vals.append("")
            else:
                coef_vals.append(latex_coef_se(
                    r["first_stage_coef"],
                    r["first_stage_se"],
                    r["first_stage_p"]
                ))
        rows.append({"label": "Excluded instrument", "vals": coef_vals})

        f_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            if r is None or pd.isna(r["first_stage_F"]):
                f_vals.append("")
            else:
                f_vals.append(f"{r['first_stage_F']:.2f}")
        rows.append({"label": "First-stage F-stat", "vals": f_vals})

        n_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            n_vals.append("" if r is None else str(int(r["nobs"])))
        rows.append({"label": "Observations", "vals": n_vals})

        r2_vals = []
        for fe_label, weight_label, _ in PANEL_COLS:
            r = _get_result(results_table, model_name, fe_label, weight_label)
            if r is None or "first_stage_r2" not in r.index or pd.isna(r["first_stage_r2"]):
                r2_vals.append("")
            else:
                r2_vals.append(f"{r['first_stage_r2']:.3f}")
        rows.append({"label": r"First-stage $R^2$", "vals": r2_vals})

    return rows


def write_baseline_iv_panel_table(results_table, sample_key, sample_title, sample_note, output_dir_for_run):
    rows = build_baseline_iv_panel_rows(results_table)
    outpath = os.path.join(output_dir_for_run, f"{sample_key}_baseline_iv_panel_table.tex")

    notes = (
        "Notes: Robust standard errors are reported in parentheses. "
        "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
        "The dependent variable is log SNAP participation rate, defined as log(SNAP persons divided by the July 1, 2025 Census resident population estimate). "
        "The baseline panel reports OLS estimates. IV panels report second-stage estimates where "
        "$\\log(\\hat{\\bar a}_s)$ is instrumented using the excluded instrument shown in each panel. "
        f"{sample_note} "
        "Crop instruments are log(1 + historical crop acres per 1,000 state land acres) averaged over 1926--1940. "
        "Suitability is average days suitable for fieldwork during April--October. "
        "Reliability weights use the post-calibration Soft-DTW matching score."
    )

    return _write_panel_table_from_rows(
        rows=rows,
        caption=f"{sample_title} Baseline and IV Estimates: SNAP Participation Rate",
        label=f"tab:{sample_key}_baseline_iv_panel",
        outpath=outpath,
        notes=notes,
    )


def write_first_stage_panel_table(results_table, sample_key, sample_title, sample_note, output_dir_for_run):
    rows = build_first_stage_panel_rows(results_table)
    outpath = os.path.join(output_dir_for_run, f"{sample_key}_first_stage_panel_table.tex")

    notes = (
        "Notes: Robust standard errors are reported in parentheses. "
        "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
        "The dependent variable in each first-stage regression is $\\log(\\hat{\\bar a}_s)$. "
        "The row labelled Excluded instrument reports the coefficient on the excluded instrument shown in each panel. "
        f"{sample_note} "
        "Crop instruments are log(1 + historical crop acres per 1,000 state land acres) averaged over 1926--1940. "
        "Suitability is average days suitable for fieldwork during April--October. "
        "Reliability weights use the post-calibration Soft-DTW matching score."
    )

    return _write_panel_table_from_rows(
        rows=rows,
        caption=f"{sample_title} First-Stage Estimates",
        label=f"tab:{sample_key}_first_stage_panel",
        outpath=outpath,
        notes=notes,
    )


# ============================================================
# ANALYSIS RUNNER
# ============================================================

def run_sample_analysis(base_reg, sample_key, sample_title, sample_note, output_dir_for_run):
    os.makedirs(output_dir_for_run, exist_ok=True)
    reg = base_reg.copy()

    region_dummies = pd.get_dummies(reg["region"], prefix="region", drop_first=True, dtype=float)
    reg = pd.concat([reg, region_dummies], axis=1)
    region_fe = list(region_dummies.columns)

    reg.to_csv(os.path.join(output_dir_for_run, f"{sample_key}_regression_dataset.csv"), index=False)

    print(f"\n{sample_title} states used:")
    print(sorted(reg["state"].unique()))
    print("N states:", reg["state"].nunique())
    print("Region FE candidates:", region_fe)

    fe_specs = [("No region FE", []), ("Region FE", region_fe)]
    results = []

    for fe_label, controls in fe_specs:
        for weight_label, weight_col, weight_display in WEIGHT_SPECS:
            # Baseline OLS/WLS using log(abar)
            try:
                ols, _ = run_ols(y=YVAR, xvars=[XVAR], controls=controls, data=reg, weight_col=weight_col)
                results.append(make_result_row(
                    model_name="Baseline", weight_label=weight_label, weight_display=weight_display,
                    fe_label=fe_label, instrument="None", model=ols, coef_name=XVAR
                ))
                with open(os.path.join(output_dir_for_run, safe_filename(f"{sample_key}_Baseline_{weight_label}_{fe_label}_summary.txt")), "w", encoding="utf-8") as f:
                    f.write(str(ols.summary()))
            except Exception as e:
                print(f"{sample_title} Baseline failed: {weight_label}, {fe_label}, {repr(e)}")

            # IV models
            for iv_label, z, _ in IV_SPECS:
                try:
                    iv, fs, _ = run_iv(y=YVAR, x=XVAR, z=z, controls=controls, data=reg, weight_col=weight_col)
                    results.append(make_result_row(
                        model_name=iv_label, weight_label=weight_label, weight_display=weight_display,
                        fe_label=fe_label, instrument=z, model=iv, coef_name=XVAR,
                        first_stage_F=iv.first_stage_F,
                        first_stage_coef=iv.first_stage_coef,
                        first_stage_se=iv.first_stage_se,
                        first_stage_p=iv.first_stage_p,
                        first_stage_r2=iv.first_stage_r2,
                    ))
                    with open(os.path.join(output_dir_for_run, safe_filename(f"{sample_key}_{iv_label}_{weight_label}_{fe_label}_summary.txt")), "w", encoding="utf-8") as f:
                        f.write("Second stage:\n")
                        f.write(str(iv.summary))
                        f.write("\n\nFirst stage:\n")
                        f.write(str(fs.summary()))
                except Exception as e:
                    print(f"{sample_title} {iv_label} failed: {weight_label}, {fe_label}, {repr(e)}")

    results_table = pd.DataFrame(results)
    if results_table.empty:
        raise ValueError(f"No regressions estimated successfully for {sample_title}.")

    results_table["stars"] = results_table["p"].apply(stars)
    results_table["first_stage_stars"] = results_table["first_stage_p"].apply(stars)
    results_table.to_csv(os.path.join(output_dir_for_run, f"{sample_key}_all_results_long_with_first_stage.csv"), index=False)

    table_specs = [
        ("Baseline", f"{sample_title} Baseline Estimates: SNAP Participation Rate", f"tab:{sample_key}_baseline", f"{sample_key}_baseline_table"),
        ("Log Cotton Acreage Intensity IV", f"{sample_title} IV Estimates: Log Cotton Acreage Intensity: SNAP Participation Rate", f"tab:{sample_key}_log_cotton_acreage_intensity_iv", f"{sample_key}_log_cotton_acreage_intensity_iv_table"),
        ("Log Corn Acreage Intensity IV", f"{sample_title} IV Estimates: Log Corn Acreage Intensity: SNAP Participation Rate", f"tab:{sample_key}_log_corn_acreage_intensity_iv", f"{sample_key}_log_corn_acreage_intensity_iv_table"),
        ("Log Soybean Acreage Intensity IV", f"{sample_title} IV Estimates: Log Soybean Acreage Intensity: SNAP Participation Rate", f"tab:{sample_key}_log_soybean_acreage_intensity_iv", f"{sample_key}_log_soybean_acreage_intensity_iv_table"),
        ("Suitability IV", f"{sample_title} IV Estimates: Suitability: SNAP Participation Rate", f"tab:{sample_key}_suitability_iv", f"{sample_key}_suitability_iv_table"),
    ]

    all_tables = {}
    for model_name, caption, label, filename_stub in table_specs:
        table = build_single_model_table(results_table, model_name)
        all_tables[model_name] = table
        table.to_csv(os.path.join(output_dir_for_run, f"{filename_stub}.csv"))
        write_latex_table_for_output(table, caption, label, f"{filename_stub}.tex", output_dir_for_run, sample_note)

    combined_tex_path = os.path.join(output_dir_for_run, f"{sample_key}_all_separate_tables_with_first_stage.tex")
    with open(combined_tex_path, "w", encoding="utf-8") as f:
        for _, _, _, filename_stub in table_specs:
            table_path = os.path.join(output_dir_for_run, f"{filename_stub}.tex")
            with open(table_path, "r", encoding="utf-8") as tf:
                f.write(tf.read())
                f.write("\n\n")

    # ------------------------------------------------------------
    # New aligned panel-style output tables:
    #   1. Baseline + IV second-stage estimates
    #   2. First-stage estimates
    # ------------------------------------------------------------

    baseline_iv_panel_path = write_baseline_iv_panel_table(
        results_table=results_table,
        sample_key=sample_key,
        sample_title=sample_title,
        sample_note=sample_note,
        output_dir_for_run=output_dir_for_run,
    )

    first_stage_panel_path = write_first_stage_panel_table(
        results_table=results_table,
        sample_key=sample_key,
        sample_title=sample_title,
        sample_note=sample_note,
        output_dir_for_run=output_dir_for_run,
    )

    combined_panel_tex_path = os.path.join(output_dir_for_run, f"{sample_key}_baseline_iv_and_first_stage_panel_tables.tex")
    with open(combined_panel_tex_path, "w", encoding="utf-8") as f:
        for table_path in [baseline_iv_panel_path, first_stage_panel_path]:
            with open(table_path, "r", encoding="utf-8") as tf:
                f.write(tf.read())
                f.write("\n\n")

    first_stage_only = results_table[results_table["instrument"] != "None"].copy()
    first_stage_cols = [
        "model", "fe_spec", "weighting", "instrument", "first_stage_coef",
        "first_stage_se", "first_stage_p", "first_stage_F", "nobs",
        "region_fe", "weight_display"
    ]
    first_stage_only[first_stage_cols].to_csv(os.path.join(output_dir_for_run, f"{sample_key}_first_stage_only_summary.csv"), index=False)

    print("\n==============================")
    print(f"SEPARATE {sample_title.upper()} TABLES WITH FIRST STAGE")
    print("==============================")
    print("\nAligned panel-style LaTeX tables:")
    print("- " + baseline_iv_panel_path)
    print("- " + first_stage_panel_path)
    print("- " + combined_panel_tex_path)

    for model_name, table in all_tables.items():
        print(f"\n--- {model_name} ---")
        print(table)

    return results_table, all_tables

# ============================================================
# MERGE FULL AVAILABLE DATASET ONCE
# ============================================================

base_reg = (
    snap
    .merge(df_abar, on="state", how="inner")
    .merge(crop_wide, on="state", how="inner")
    .merge(suit[["state", SUIT]], on="state", how="inner")
)
base_reg["region"] = base_reg["state"].map(abbr_to_region)

needed_base = ["region", YVAR, XVAR, MATCH, SUIT] + [f"log_{crop}_acres" for crop in CROPS]
base_reg = base_reg.dropna(subset=needed_base).copy()
base_reg.to_csv(os.path.join(output_dir, "full_state_regression_dataset.csv"), index=False)

# ============================================================
# RUN SELECTED SAMPLE: MATCHING SCORE > 95
# ============================================================

selected_gt95_states = sorted(base_reg.loc[base_reg[MATCH] > SELECTED_MATCH_THRESHOLD, "state"].dropna().unique())
if len(selected_gt95_states) == 0:
    raise ValueError(f"No states satisfy matching_score_percent > {SELECTED_MATCH_THRESHOLD}.")

selected_gt95_reg = base_reg[base_reg["state"].isin(selected_gt95_states)].copy()
selected_output_dir = os.path.join(output_dir, "selected_gt95_log_snap_rate_log_abar_log_crop")

selected_results_table, selected_all_tables = run_sample_analysis(
    base_reg=selected_gt95_reg,
    sample_key="selected_gt95_log_snap_rate_log_abar_log_crop",
    sample_title="Selected-State, Match Score $>$ 95\\%",
    sample_note="The selected sample includes states with post-calibration Soft-DTW matching scores greater than 95 percent.",
    output_dir_for_run=selected_output_dir,
)

selected_results_table.to_csv(os.path.join(output_dir, "selected_gt95_log_snap_rate_log_abar_log_crop_all_results_long_with_first_stage.csv"), index=False)
selected_results_table[selected_results_table["instrument"] != "None"].to_csv(os.path.join(output_dir, "selected_gt95_log_snap_rate_log_abar_log_crop_first_stage_only_summary.csv"), index=False)
pd.DataFrame({"state": selected_gt95_states}).to_csv(os.path.join(output_dir, "selected_gt95_log_snap_rate_log_abar_log_crop_states_used.csv"), index=False)


# ============================================================
# COMPACT SOYBEAN IV ROBUSTNESS ACROSS MATCHING-SCORE THRESHOLDS
# ============================================================
# Only the manuscript-facing soybean IV robustness table is generated here.
# This avoids exporting exploratory/full-model threshold tables.

ROBUSTNESS_THRESHOLDS = [85.0, 90.0, 95.0]


def write_compact_soybean_threshold_table(base_reg, output_dir_for_run):
    rows = []
    for threshold in ROBUSTNESS_THRESHOLDS:
        reg0 = base_reg[base_reg[MATCH] > threshold].copy()
        if reg0.empty:
            continue
        region_dummies = pd.get_dummies(reg0["region"], prefix="region", drop_first=True, dtype=float)
        reg0 = pd.concat([reg0, region_dummies], axis=1)
        region_fe = list(region_dummies.columns)

        for weight_label, weight_col, _ in WEIGHT_SPECS:
            try:
                iv, fs, used = run_iv(
                    y=YVAR,
                    x=XVAR,
                    z="log_soybean_acres",
                    controls=region_fe,
                    data=reg0,
                    weight_col=weight_col,
                )
                if hasattr(iv, "std_errors"):
                    second_stage_se = iv.std_errors[XVAR]
                else:
                    second_stage_se = iv.bse[XVAR]
                rows.append({
                    "threshold": threshold,
                    "weighting": weight_label,
                    "coef": iv.params[XVAR],
                    "se": second_stage_se,
                    "p": iv.pvalues[XVAR],
                    "first_stage_F": iv.first_stage_F,
                    "nobs": int(iv.nobs),
                })
            except Exception as e:
                print(f"Soybean threshold robustness failed: threshold={threshold}, weight={weight_label}, {repr(e)}")

    soy = pd.DataFrame(rows)
    if soy.empty:
        raise ValueError("No soybean threshold robustness estimates were generated.")

    soy.to_csv(os.path.join(output_dir_for_run, "soybean_iv_threshold_robustness_compact_results.csv"), index=False)

    weight_order = ["Unweighted", "WLS", "WLS Sq."]
    threshold_order = [85.0, 90.0, 95.0]

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Soybean IV Robustness Across Calibration-Quality Thresholds}")
    lines.append(r"\label{tab:soybean_iv_threshold_robustness}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{adjustbox}{max width=0.92\textwidth}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline\hline")
    lines.append(r"Sample & Weighting & $\log(\hat{\bar a}_s)$ & First-stage F-stat & Observations \\")
    lines.append(r"\hline")

    for ti, threshold in enumerate(threshold_order):
        if ti > 0:
            lines.append(r"\hline")
        for weight in weight_order:
            row = soy[(soy["threshold"] == threshold) & (soy["weighting"] == weight)]
            if row.empty:
                continue
            r = row.iloc[0]
            st = stars(r["p"])
            coef_cell = f"${float(r['coef']):.3f}^{{{st}}}$" if st else f"${float(r['coef']):.3f}$"
            se_cell = f"$({float(r['se']):.3f})$"
            lines.append(rf"Match score $>{int(threshold)}\%$ ")
            lines.append(rf"& {weight} ")
            lines.append(rf"& {coef_cell} ")
            lines.append(rf"& {float(r['first_stage_F']):.2f} ")
            lines.append(rf"& {int(r['nobs'])} \\")
            lines.append(r"&  ")
            lines.append(rf"& {se_cell} ")
            lines.append(r"&  ")
            lines.append(r"&  \\")

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\vspace{0.25em}")
    lines.append(r"\begin{minipage}{0.92\textwidth}")
    lines.append(r"\footnotesize")
    lines.append(
        r"\emph{Notes:} Robust standard errors are reported in parentheses. "
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
        r"The dependent variable is log SNAP participation rate, defined as the logarithm of SNAP participants from the latest available SNAP column divided by the July 1, 2025 Census resident population estimate. "
        r"The endogenous regressor is $\log(\hat{\bar a}_s)$. "
        r"The excluded instrument is the logarithm of one plus average historical soybean acres per 1,000 state land acres over 1926--1940. "
        r"All specifications include region fixed effects defined using standard U.S. Census regional classifications. "
        r"Reliability weights use the post-calibration Soft-DTW matching score."
    )
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    outpath = os.path.join(output_dir_for_run, "soybean_iv_threshold_robustness_compact_table.tex")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return outpath


write_compact_soybean_threshold_table(base_reg=base_reg, output_dir_for_run=str(MANUSCRIPT_REGRESSION_DIR))

# ============================================================
# KEEP ONLY MANUSCRIPT-FACING REGRESSION OUTPUTS FROM THIS SCRIPT
# ============================================================
# Tables kept:
#   1. main_regression_table.tex
#   2. first_stage_table.tex
#   3. soybean_iv_threshold_robustness_compact_table.tex
# The interaction table is generated by 04_interaction_regressions.py.

main_source = os.path.join(
    selected_output_dir,
    "selected_gt95_log_snap_rate_log_abar_log_crop_baseline_iv_panel_table.tex"
)
main_target = str(MANUSCRIPT_REGRESSION_DIR / "main_regression_table.tex")
if os.path.exists(main_source):
    shutil.copyfile(main_source, main_target)
else:
    raise FileNotFoundError(f"Expected main regression table not found: {main_source}")

first_stage_source = os.path.join(
    selected_output_dir,
    "selected_gt95_log_snap_rate_log_abar_log_crop_first_stage_panel_table.tex"
)
first_stage_target = str(MANUSCRIPT_REGRESSION_DIR / "first_stage_table.tex")
if os.path.exists(first_stage_source):
    shutil.copyfile(first_stage_source, first_stage_target)
else:
    raise FileNotFoundError(f"Expected first-stage table not found: {first_stage_source}")

keep_files = {
    os.path.abspath(main_target),
    os.path.abspath(first_stage_target),
    os.path.abspath(str(MANUSCRIPT_REGRESSION_DIR / "soybean_iv_threshold_robustness_compact_table.tex")),
}
# Keep outputs/regression/ limited to the three manuscript-facing tables generated
# by this script. The interaction table is generated later by 04_interaction_regressions.py.
for fname in os.listdir(MANUSCRIPT_REGRESSION_DIR):
    path = os.path.abspath(os.path.join(MANUSCRIPT_REGRESSION_DIR, fname))
    if os.path.isfile(path) and path not in keep_files:
        os.remove(path)

print("\nFinal manuscript-facing regression outputs from 03_main_regressions.py:")
for path in sorted(p for p in keep_files if os.path.exists(p)):
    print(path)
