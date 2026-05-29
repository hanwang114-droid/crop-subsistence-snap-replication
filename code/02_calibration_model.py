# -----------------------------------------------------------------------------
# This script is part of the replication workflow. It assumes that
# code/00_clean_data.py has already converted original files in raw_data/ into
# clean_data/. This script reads cleaned inputs only; it does not read raw_data/.
# -----------------------------------------------------------------------------

"""Calibrate the structural-transformation subsistence index.

This script is the source of the paper's calibrated subsistence index. It uses
raw USDA--ERS state agricultural labor productivity, defined as agricultural
output quantity divided by labor input quantity. It deliberately does NOT use
the old within-state 1960-normalized productivity series.

Baseline calibration:
  * early calibration moments: 1960 and 1970.

Robustness calibrations:
  * 1960 only;
  * 1960, 1970, and 1980.

For each candidate common alpha, the code recovers a state-specific abar_s from
the indicated early-year moments, then chooses alpha by minimizing full-panel
squared prediction error. Soft-DTW matching scores are computed after calibration
as diagnostics and reliability weights; they are not used to choose alpha or
abar_s.

Final manuscript-facing files are written to outputs/calibration/. Intermediate
CSV files required by the regression scripts are written to derived/calibration/.
"""
# -----------------------------------------------------------------------------
# READER GUIDE
# -----------------------------------------------------------------------------
# This file produces the calibrated subsistence index used in the regressions.
# The key choices are: use raw state labor productivity, use 1960 and 1970 as
# the baseline early calibration years, and report two robustness calibrations.
# The outputs in derived/calibration are used by the regression scripts; the
# outputs in outputs/calibration are the tables/figures for the manuscript.
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from tslearn.metrics import soft_dtw
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    TSLEARN_AVAILABLE = True
except Exception:
    TSLEARN_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "clean_data" / "structrual_transformation_data"
DERIVED_DIR = ROOT / "derived" / "calibration"
OUT_DIR = ROOT / "outputs" / "calibration"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Required folder: one model-vs-data calibration figure for every calibrated state.
STATE_MODEL_FIT_DIR = OUT_DIR / "state_model_fit_all"

PANEL = DATA_DIR / "calibration_state_year_panel.csv"

STATE_COL = "state"
YEAR_COL = "year"
SHARE_COL = "ag_share_ag_forest_fish"
PROD_COL = "Aa_labor_productivity_index"

BASELINE_EARLY_YEARS = [1960, 1970]
ROBUSTNESS_EARLY_YEAR_SPECS = {
    "1960 only": [1960],
    "Baseline: 1960, 1970": [1960, 1970],
    "1960, 1970, 1980": [1960, 1970, 1980],
}
MIN_EARLY_OBS = 1
ALPHA_LOWER = 1e-8
ALPHA_UPPER_DEFAULT = 0.50
ALLOW_NEGATIVE_ABAR = False
OPTIMIZER_XATOL = 1e-6
SELECTED_MATCH_THRESHOLD = 90.0
SOFT_DTW_GAMMA = 0.1

# Derived files used downstream by regressions.
PARAM_FILE = DERIVED_DIR / "calibrated_state_abar_optimized_alpha_raw_productivity.csv"
SIM_FILE = DERIVED_DIR / "simulation_state_time_series_optimized_alpha_raw_productivity.csv"
SUMMARY_FILE = DERIVED_DIR / "calibration_summary_optimized_alpha_raw_productivity.csv"
MATCHING_FILE = DERIVED_DIR / "soft_dtw_matching_quality_by_state_raw_productivity.csv"
SELECTED_FILE = DERIVED_DIR / "selected_states_matching_score_gt90_raw_productivity.csv"
COMPAT_PARAM_FILE = DERIVED_DIR / "calibrated_state_abar_common_alpha_raw_productivity.csv"
COMPAT_SIM_FILE = DERIVED_DIR / "simulation_state_time_series_common_alpha_raw_productivity.csv"
ROBUSTNESS_SUMMARY_FILE = DERIVED_DIR / "calibration_early_year_robustness_summary_raw_productivity.csv"

# Manuscript-facing files.
LATEX_SUMMARY_FILE = OUT_DIR / "main_calibration_summary_table_raw_productivity.tex"
LATEX_COMPARISON_FILE = OUT_DIR / "main_calibration_results_early_year_robustness_raw_productivity.tex"
FIG_AVG = OUT_DIR / "figure_average_data_vs_model_raw_productivity.png"
FIG_ABAR = OUT_DIR / "figure_abar_distribution_raw_productivity.png"
FIG_SCATTER = OUT_DIR / "figure_model_vs_data_scatter_raw_productivity.png"
FIG_SCORE = OUT_DIR / "figure_soft_dtw_matching_score_by_state_raw_productivity.png"
FIG_ALPHA_OBJECTIVE = OUT_DIR / "figure_alpha_objective_raw_productivity.png"


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    required = [STATE_COL, YEAR_COL, SHARE_COL, PROD_COL]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")
    if PROD_COL == "Aa_labor_productivity_index_1960_state1":
        raise ValueError("Raw-productivity calibration must not use the 1960-normalized series.")

    d = raw.copy()
    d[STATE_COL] = d[STATE_COL].astype(str).str.upper().str.strip()
    for col in [YEAR_COL, SHARE_COL, PROD_COL]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[STATE_COL, YEAR_COL, SHARE_COL, PROD_COL]).copy()
    d = d[(d[SHARE_COL] >= 0) & (d[SHARE_COL] <= 1) & (d[PROD_COL] > 0)].copy()
    d = d.rename(columns={STATE_COL: "state", YEAR_COL: "year", SHARE_COL: "data_ag_share", PROD_COL: "Aa"})
    d["year"] = d["year"].astype(int)
    d = d.sort_values(["state", "year"]).drop_duplicates(["state", "year"], keep="first")
    return d[["state", "year", "data_ag_share", "Aa"]]


def _early_table(d: pd.DataFrame, early_years: list[int]) -> pd.DataFrame:
    early = d[d["year"].isin(early_years)].copy()
    n_early = early.groupby("state").size()
    keep = n_early[n_early >= MIN_EARLY_OBS].index
    early = early[early["state"].isin(keep)].copy()
    return early


def _abar_by_state(d: pd.DataFrame, alpha: float, early_years: list[int]) -> pd.Series:
    early = _early_table(d, early_years)
    if early.empty or abs(1 - alpha) < 1e-12:
        return pd.Series(dtype=float)
    early = early.copy()
    early["abar_each"] = early["Aa"] * (early["data_ag_share"] - alpha) / (1 - alpha)
    abar = early.groupby("state")["abar_each"].mean()
    abar = abar.replace([np.inf, -np.inf], np.nan).dropna()
    if not ALLOW_NEGATIVE_ABAR:
        abar = abar[abar > 0]
    return abar


def simulate(d: pd.DataFrame, alpha: float, early_years: list[int]):
    abar = _abar_by_state(d, alpha, early_years)
    if abar.empty:
        return pd.DataFrame(), pd.DataFrame()
    sim = d[d["state"].isin(abar.index)].copy()
    sim["alpha_common"] = alpha
    sim["abar_state"] = sim["state"].map(abar)
    sim["model_ag_share"] = alpha + (1 - alpha) * sim["abar_state"] / sim["Aa"]
    sim["residual"] = sim["data_ag_share"] - sim["model_ag_share"]
    sim["model_ag_share_clipped"] = sim["model_ag_share"].clip(0, 1)
    sim["residual_clipped"] = sim["data_ag_share"] - sim["model_ag_share_clipped"]
    sim["data_nonag_share"] = 1 - sim["data_ag_share"]
    sim["model_nonag_share"] = 1 - sim["model_ag_share"]

    rows = []
    for state, g in sim.groupby("state"):
        early = g[g["year"].isin(early_years)]
        untargeted = g[~g["year"].isin(early_years)]
        rows.append({
            "state": state,
            "alpha_common": alpha,
            "abar_state": float(abar.loc[state]),
            "early_years_for_abar": ", ".join(map(str, early_years)),
            "n_early_obs_used_for_abar": int(len(early)),
            "mean_early_ag_share": float(early["data_ag_share"].mean()) if len(early) else np.nan,
            "mean_early_Aa": float(early["Aa"].mean()) if len(early) else np.nan,
            "min_year": int(g["year"].min()),
            "max_year": int(g["year"].max()),
            "n_years_simulated": int(len(g)),
            "rmse_all_years": float(np.sqrt(np.nanmean(g["residual"] ** 2))),
            "mae_all_years": float(np.nanmean(np.abs(g["residual"]))),
            "rmse_untargeted_years": float(np.sqrt(np.nanmean(untargeted["residual"] ** 2))) if len(untargeted) else np.nan,
            "max_abs_error": float(np.nanmax(np.abs(g["residual"]))),
        })
    return pd.DataFrame(rows), sim.reset_index(drop=True)


def alpha_loss(alpha: float, d: pd.DataFrame, early_years: list[int]) -> float:
    """Fast full-panel SSE objective without constructing output tables."""
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        return 1e30
    abar = _abar_by_state(d, alpha, early_years)
    if abar.empty:
        return 1e30
    dd = d[d["state"].isin(abar.index)].copy()
    aa = dd["state"].map(abar).to_numpy(dtype=float)
    pred = alpha + (1 - alpha) * aa / dd["Aa"].to_numpy(dtype=float)
    resid = dd["data_ag_share"].to_numpy(dtype=float) - pred
    return float(np.nansum(resid ** 2))


def optimize_alpha(d: pd.DataFrame, early_years: list[int]):
    early = _early_table(d, early_years)
    if early.empty:
        raise ValueError(f"No early observations for {early_years}")
    if not ALLOW_NEGATIVE_ABAR:
        upper = min(ALPHA_UPPER_DEFAULT, float(early.groupby("state")["data_ag_share"].min().min()) * 0.999)
    else:
        upper = ALPHA_UPPER_DEFAULT
    if upper <= ALPHA_LOWER:
        raise ValueError("Alpha upper bound is not above lower bound.")

    # Use the same bounded scalar optimizer as the original script, but with a fast vectorized loss.
    if SCIPY_AVAILABLE:
        res = minimize_scalar(lambda a: alpha_loss(float(a), d, early_years), bounds=(ALPHA_LOWER, upper), method="bounded", options={"xatol": OPTIMIZER_XATOL})
        alpha_hat = float(res.x)
    else:
        grid_vals_for_opt = np.linspace(ALPHA_LOWER, upper, 400)
        losses = [alpha_loss(a, d, early_years) for a in grid_vals_for_opt]
        alpha_hat = float(grid_vals_for_opt[int(np.argmin(losses))])

    grid_vals = np.linspace(ALPHA_LOWER, upper, 120)
    grid = []
    for a in grid_vals:
        sse = alpha_loss(float(a), d, early_years)
        abar = _abar_by_state(d, float(a), early_years)
        nstates = int(len(abar))
        nobs = int(d[d["state"].isin(abar.index)].shape[0]) if nstates else 0
        grid.append({"alpha": float(a), "sse": sse if sse < 1e29 else np.nan, "rmse": float(np.sqrt(sse / nobs)) if nobs and sse < 1e29 else np.nan, "n_obs": nobs, "n_states": nstates})
    return alpha_hat, pd.DataFrame(grid)


def safe_corr(x, y):
    if len(x) < 2 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y)))


def _scale_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return ((x - np.nanmean(x)) / sd).reshape(-1, 1)


def soft_dtw_fallback(x: np.ndarray, y: np.ndarray, gamma: float = 0.1) -> float:
    """Small dependency-free Soft-DTW implementation for 1D trajectories.

    This fallback follows the standard Cuturi--Blondel dynamic program and is
    used only when tslearn is unavailable. It keeps the replication package
    portable while preserving the same matching-score concept.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n, m = len(x), len(y)
    D = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    R = np.full((n + 2, m + 2), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            vals = np.array([-R[i - 1, j - 1] / gamma, -R[i - 1, j] / gamma, -R[i, j - 1] / gamma])
            vmax = np.max(vals)
            softmin = -gamma * (np.log(np.exp(vals - vmax).sum()) + vmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return float(R[n, m])


def compute_matching(simulation: pd.DataFrame, params: pd.DataFrame):
    scaler = TimeSeriesScalerMeanVariance() if TSLEARN_AVAILABLE else None
    if not TSLEARN_AVAILABLE:
        warnings.warn("tslearn not available; using built-in Soft-DTW fallback.")
    rows = []
    for state, g in simulation.sort_values(["state", "year"]).groupby("state"):
        data = g["data_ag_share"].to_numpy(float)
        model = g["model_ag_share"].to_numpy(float)
        residual = data - model
        if len(data) >= 2:
            try:
                if TSLEARN_AVAILABLE:
                    data_scaled = scaler.fit_transform(data.reshape(1, -1, 1))[0]
                    model_scaled = scaler.fit_transform(model.reshape(1, -1, 1))[0]
                    dist = float(soft_dtw(data_scaled, model_scaled, gamma=SOFT_DTW_GAMMA))
                else:
                    data_scaled = _scale_series(data)
                    model_scaled = _scale_series(model)
                    dist = soft_dtw_fallback(data_scaled, model_scaled, gamma=SOFT_DTW_GAMMA)
            except Exception:
                dist = np.nan
        else:
            dist = np.nan
        rows.append({
            "state": state,
            "alpha_common": float(params.loc[params["state"] == state, "alpha_common"].iloc[0]),
            "abar_state": float(params.loc[params["state"] == state, "abar_state"].iloc[0]),
            "mean_data_ag_share": float(np.nanmean(data)),
            "mean_model_ag_share": float(np.nanmean(model)),
            "mean_residual": float(np.nanmean(residual)),
            "rmse": float(np.sqrt(np.nanmean(residual ** 2))),
            "mae": float(np.nanmean(np.abs(residual))),
            "mape": float(np.nanmean(np.abs(residual / np.maximum(data, 1e-8)))),
            "correlation": safe_corr(data, model),
            "soft_dtw_distance": dist,
            "max_abs_error": float(np.nanmax(np.abs(residual))),
            "n_years_compared": int(len(g)),
        })
    matching = pd.DataFrame(rows)
    valid = matching["soft_dtw_distance"].replace([np.inf, -np.inf], np.nan)
    mn, mx = valid.min(), valid.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        matching["matching_score_percent"] = 100.0
    else:
        matching["matching_score_percent"] = (100 * (1 - (matching["soft_dtw_distance"] - mn) / (mx - mn))).clip(0, 100)
    matching = matching.sort_values("matching_score_percent", ascending=False).reset_index(drop=True)
    matching["fit_rank"] = np.arange(1, len(matching) + 1)
    return matching


def make_summary(alpha, params, sim, matching, early_years):
    rows = [
        ("model", "Stone-Geary non-CES"),
        ("calibration_method", "optimized_common_alpha_full_panel_sse_raw_productivity"),
        ("productivity_variable", PROD_COL),
        ("optimized_alpha", alpha),
        ("optimizer_xatol", OPTIMIZER_XATOL),
        ("early_years_for_abar", ", ".join(map(str, early_years))),
        ("allow_negative_abar", ALLOW_NEGATIVE_ABAR),
        ("n_states_calibrated", params["state"].nunique()),
        ("n_state_year_observations", len(sim)),
        ("min_year", sim["year"].min()),
        ("max_year", sim["year"].max()),
        ("mean_abar", params["abar_state"].mean()),
        ("median_abar", params["abar_state"].median()),
        ("min_abar", params["abar_state"].min()),
        ("max_abar", params["abar_state"].max()),
        ("mean_rmse_all_years", params["rmse_all_years"].mean()),
        ("median_rmse_all_years", params["rmse_all_years"].median()),
        ("mean_rmse_untargeted_years", params["rmse_untargeted_years"].mean()),
        ("mean_matching_score", matching["matching_score_percent"].mean()),
        ("median_matching_score", matching["matching_score_percent"].median()),
        ("selected_match_threshold", SELECTED_MATCH_THRESHOLD),
        ("n_selected_states", int((matching["matching_score_percent"] > SELECTED_MATCH_THRESHOLD).sum())),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def run_calibration(d, early_years):
    alpha, grid = optimize_alpha(d, early_years)
    params, sim = simulate(d, alpha, early_years)
    matching = compute_matching(sim, params)
    selected = matching[matching["matching_score_percent"] > SELECTED_MATCH_THRESHOLD].copy()
    summary = make_summary(alpha, params, sim, matching, early_years)
    return {"early_years": early_years, "alpha": alpha, "grid": grid, "params": params, "sim": sim, "matching": matching, "selected": selected, "summary": summary}


def sv(run, key):
    s = run["summary"]
    r = s.loc[s["statistic"] == key, "value"]
    return r.iloc[0] if not r.empty else np.nan


def ff(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fi(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def write_latex_tables(runs):
    baseline = runs["Baseline: 1960, 1970"]
    # One-column baseline summary, kept for compatibility with earlier manuscript references.
    b = baseline
    latex1 = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Main Calibration Summary}}
\label{{tab:main_calibration_summary}}
\small
\setlength{{\tabcolsep}}{{6pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{adjustbox}}{{max width=0.80\textwidth}}
\begin{{tabular}}{{lc}}
\hline\hline
Object & Value \\
\hline
Calibration method & Full-panel optimized common $\alpha$ using raw productivity \\
Common $\hat\alpha$ & {ff(sv(b, 'optimized_alpha'), 4)} \\
Productivity variable & {PROD_COL} \\
Early years for $\bar a_s$ & {sv(b, 'early_years_for_abar')} \\
States calibrated & {fi(sv(b, 'n_states_calibrated'))} \\
State-year observations & {fi(sv(b, 'n_state_year_observations'))} \\
Mean $\bar a_s$ & {ff(sv(b, 'mean_abar'), 4)} \\
Median $\bar a_s$ & {ff(sv(b, 'median_abar'), 4)} \\
Mean RMSE & {ff(sv(b, 'mean_rmse_all_years'), 4)} \\
Mean Soft-DTW matching score & {ff(sv(b, 'mean_matching_score'), 2)} \\
Selected states & {fi(sv(b, 'n_selected_states'))} \\
\hline\hline
\end{{tabular}}
\end{{adjustbox}}
\vspace{{0.25em}}
\begin{{minipage}}{{0.80\textwidth}}
\footnotesize
\emph{{Notes:}} The table summarizes the baseline raw-productivity Stone--Geary calibration. Productivity is not normalized to one in 1960. The common parameter $\alpha$ is chosen to minimize full-panel squared prediction error. Soft-DTW matching scores are computed after calibration and are not used to choose $\alpha$ or $\bar a_s$.
\end{{minipage}}
\end{{table}}
"""
    LATEX_SUMMARY_FILE.write_text(latex1, encoding="utf-8")

    order = [("1960 only", runs["1960 only"]), ("Baseline: 1960, 1970", runs["Baseline: 1960, 1970"]), ("1960, 1970, 1980", runs["1960, 1970, 1980"])]
    def vals(key, kind="float", digits=4):
        out = []
        for _, run in order:
            v = sv(run, key)
            out.append(fi(v) if kind == "int" else str(v) if kind == "str" else ff(v, digits))
        return out
    rows = [
        (r"\multicolumn{4}{l}{\textbf{Calibrated Parameters}}", None),
        (r"Common preference parameter, $\hat\alpha$", vals("optimized_alpha", digits=4)),
        (r"Mean calibrated subsistence index, $\hat{\bar a}_s$", vals("mean_abar", digits=4)),
        (r"Median calibrated subsistence index, $\hat{\bar a}_s$", vals("median_abar", digits=4)),
        ("Early calibration years", vals("early_years_for_abar", kind="str")),
        ("MIDRULE", None),
        (r"\multicolumn{4}{l}{\textbf{Matching Results}}", None),
        ("Mean Soft-DTW matching score", vals("mean_matching_score", digits=2)),
        ("Median Soft-DTW matching score", vals("median_matching_score", digits=2)),
        ("Mean RMSE", vals("mean_rmse_all_years", digits=4)),
        (r"Selected states, matching score $>90\%$", vals("n_selected_states", kind="int")),
        ("MIDRULE", None),
        (r"\multicolumn{4}{l}{\textbf{Sample and Estimation}}", None),
        ("Calibrated states", vals("n_states_calibrated", kind="int")),
        ("State-year observations", vals("n_state_year_observations", kind="int")),
    ]
    lb = '\\\\'
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Main Calibration Results and Early-Year Robustness}",
        r"\label{tab:main_calibration_results}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        f"Object & 1960 only & Baseline: 1960, 1970 & 1960, 1970, 1980 {lb}",
        r"\hline",
    ]
    for label, row in rows:
        if label == "MIDRULE":
            lines.append(r"\hline")
        elif row is None:
            lines.append(f"{label} {lb}")
        else:
            lines.append(label + " & " + " & ".join(row) + f" {lb}")
    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize",
        r"\emph{Notes:} The calibration uses raw USDA--ERS state labor productivity, defined as agricultural output quantity divided by labor input quantity, rather than a within-state 1960-normalized productivity series. The baseline calibration recovers $\bar a_s(\alpha)$ using the 1960 and 1970 calibration moments. The adjacent columns report robustness checks using only 1960 and using 1960, 1970, and 1980. In all columns, the common parameter $\hat\alpha$ is chosen to minimize full-panel squared prediction error. Conditional on each candidate $\alpha$, $\bar a_s(\alpha)$ is recovered analytically using the indicated early calibration moments and the corresponding raw productivity values. Soft-DTW is computed only after calibration.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    LATEX_COMPARISON_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame([{ "specification": k, "early_years": ", ".join(map(str, v["early_years"])), "alpha_hat": v["alpha"], "mean_abar": sv(v, "mean_abar"), "median_abar": sv(v, "median_abar"), "mean_matching_score": sv(v, "mean_matching_score"), "median_matching_score": sv(v, "median_matching_score"), "mean_rmse_all_years": sv(v, "mean_rmse_all_years"), "n_selected_states": sv(v, "n_selected_states"), "n_states_calibrated": sv(v, "n_states_calibrated"), "n_state_year_observations": sv(v, "n_state_year_observations") } for k,v in runs.items()]).to_csv(ROBUSTNESS_SUMMARY_FILE, index=False)


def plot_figures(run):
    sim, params, matching, grid = run["sim"], run["params"], run["matching"], run["grid"]
    avg = sim.groupby("year")[["data_ag_share", "model_ag_share"]].mean().reset_index()
    plt.figure(figsize=(8,5)); plt.plot(avg.year, avg.data_ag_share, marker="o", label="Data"); plt.plot(avg.year, avg.model_ag_share, marker="s", label="Model"); plt.xlabel("Year"); plt.ylabel("Agricultural employment share"); plt.title("Average Agricultural Share: Data vs Raw-Productivity Stone-Geary Model"); plt.legend(); plt.savefig(FIG_AVG, dpi=300, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8,5)); plt.hist(params["abar_state"].dropna(), bins=15); plt.xlabel("State-specific subsistence index, abar_s"); plt.ylabel("Number of states"); plt.title("Distribution of Calibrated Subsistence Indices"); plt.savefig(FIG_ABAR, dpi=300, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6,6)); plt.scatter(sim["data_ag_share"], sim["model_ag_share"]); mx=max(sim["data_ag_share"].max(), sim["model_ag_share"].max()); mn=min(sim["data_ag_share"].min(), sim["model_ag_share"].min(), 0); plt.plot([mn,mx],[mn,mx]); plt.xlabel("Data agricultural share"); plt.ylabel("Model agricultural share"); plt.title("Model vs Data: All State-Year Observations"); plt.savefig(FIG_SCATTER, dpi=300, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(10,8)); p=matching.sort_values("matching_score_percent"); plt.barh(p["state"], p["matching_score_percent"]); plt.xlabel("Soft-DTW Relative Matching Score (%)"); plt.ylabel("State"); plt.title("Post-Calibration Matching Score by State"); plt.savefig(FIG_SCORE, dpi=300, bbox_inches="tight"); plt.close()
    valid = grid.dropna(subset=["rmse"]); plt.figure(figsize=(8,5)); plt.plot(valid["alpha"], valid["rmse"]); plt.xlabel("Common alpha"); plt.ylabel("Full-panel RMSE"); plt.title("Alpha Calibration Objective"); plt.savefig(FIG_ALPHA_OBJECTIVE, dpi=300, bbox_inches="tight"); plt.close()



def plot_all_state_model_fit_figures(simulation: pd.DataFrame, output_dir: Path):
    """Create one calibrated model-vs-data figure for every calibrated state.

    This is a required replication output, not an optional diagnostic. Each figure
    compares the observed agricultural employment share with the calibrated model
    prediction for a single state. The output folder is:
        outputs/calibration/state_model_fit_all/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"state", "year", "data_ag_share", "model_ag_share"}
    missing = required_cols - set(simulation.columns)
    if missing:
        raise ValueError(f"Cannot draw state model-fit figures. Missing columns: {sorted(missing)}")

    states = sorted(simulation["state"].dropna().unique())
    if not states:
        raise ValueError("Cannot draw state model-fit figures because no calibrated states are available.")

    for state in states:
        g = simulation[simulation["state"] == state].sort_values("year").copy()
        if g.empty:
            continue

        plt.figure(figsize=(7, 4.5))
        plt.plot(g["year"], g["data_ag_share"], marker="o", linewidth=1.5, label="Data")
        plt.plot(g["year"], g["model_ag_share"], marker="s", linewidth=1.5, label="Model")
        plt.xlabel("Year")
        plt.ylabel("Agricultural employment share")
        plt.title(f"{state}: Data vs Calibrated Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{state}_data_vs_model.png", dpi=300, bbox_inches="tight")
        plt.close()

    # A small manifest makes it easy for readers to confirm all figures were created.
    manifest = pd.DataFrame({"state": states, "figure": [f"{s}_data_vs_model.png" for s in states]})
    manifest.to_csv(output_dir / "state_model_fit_manifest.csv", index=False)
    print(f"Saved {len(states)} state-level model-fit figures to: {output_dir}")


def main():
    print("Running raw-productivity calibration with early-year robustness...")
    d = prepare_panel(pd.read_csv(PANEL))
    early_check = d[d["year"].isin([1960, 1970, 1980])]
    print("Early-period raw productivity summary:")
    print(early_check["Aa"].describe())
    runs = {}
    suffix = {"1960 only": "early1960", "Baseline: 1960, 1970": "baseline_1960_1970", "1960, 1970, 1980": "early1960_1970_1980"}
    for label, years in ROBUSTNESS_EARLY_YEAR_SPECS.items():
        print(f"  {label}: {years}")
        run = run_calibration(d, years)
        runs[label] = run
        run["params"].to_csv(DERIVED_DIR / f"calibrated_state_abar_raw_productivity_{suffix[label]}.csv", index=False)
        run["sim"].to_csv(DERIVED_DIR / f"simulation_state_time_series_raw_productivity_{suffix[label]}.csv", index=False)
        run["matching"].to_csv(DERIVED_DIR / f"soft_dtw_matching_quality_raw_productivity_{suffix[label]}.csv", index=False)
        run["summary"].to_csv(DERIVED_DIR / f"calibration_summary_raw_productivity_{suffix[label]}.csv", index=False)
    b = runs["Baseline: 1960, 1970"]
    b["params"].to_csv(PARAM_FILE, index=False); b["sim"].to_csv(SIM_FILE, index=False); b["summary"].to_csv(SUMMARY_FILE, index=False); b["matching"].to_csv(MATCHING_FILE, index=False); b["selected"].to_csv(SELECTED_FILE, index=False); b["params"].to_csv(COMPAT_PARAM_FILE, index=False); b["sim"].to_csv(COMPAT_SIM_FILE, index=False)
    write_latex_tables(runs)

    # Required calibration graphics. These are not optional diagnostics:
    # they are produced every time the replication workflow is run.
    plot_figures(b)
    plot_all_state_model_fit_figures(b["sim"], STATE_MODEL_FIT_DIR)

    print("Calibration outputs written to", OUT_DIR)
    print("Derived calibration CSVs written to", DERIVED_DIR)


if __name__ == "__main__":
    main()
