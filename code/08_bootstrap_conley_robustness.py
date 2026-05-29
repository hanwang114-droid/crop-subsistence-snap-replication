"""Bootstrap and Conley-style robustness for the soybean IV.

This script reads the full state regression dataset written by
code/03_main_regressions.py and focuses on the soybean acreage-intensity IV.
It keeps the robustness exercise separate from the main table so the headline
specification remains easy to audit.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived" / "regression_work"
OUT = ROOT / "outputs" / "regression"
OUT.mkdir(parents=True, exist_ok=True)

DATA = DERIVED / "full_state_regression_dataset.csv"

Y = "log_snap_rate"
X = "log_abar_state"
Z = "log_soybean_acres"
MATCH = "matching_score_percent"

THRESHOLDS = [85.0, 90.0, 95.0]
WEIGHTS = [
    ("Unweighted", None),
    ("WLS", "match_weight"),
    ("WLS Sq.", "match_weight_sq"),
]
BOOT_REPS = 999
RANDOM_SEED = 20260526
CONLEY_SHARES = [0.10, 0.25, 0.50]


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def region_controls(df: pd.DataFrame) -> list[str]:
    dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True, dtype=float)
    for col in dummies.columns:
        df[col] = dummies[col].to_numpy()
    return list(dummies.columns)


def keep_full_rank_controls(df: pd.DataFrame, controls: list[str]) -> list[str]:
    kept: list[str] = []
    xmat = np.ones((len(df), 1))
    rank = np.linalg.matrix_rank(xmat)
    for col in controls:
        trial = np.column_stack([xmat, df[[col]].to_numpy(dtype=float)])
        trial_rank = np.linalg.matrix_rank(trial)
        if trial_rank > rank:
            kept.append(col)
            xmat = trial
            rank = trial_rank
    return kept


def estimation_sample(df: pd.DataFrame, threshold: float, weight_col: str | None) -> tuple[pd.DataFrame, list[str]]:
    sample = df[df[MATCH] > threshold].copy()
    controls = region_controls(sample)
    needed = [Y, X, Z, "region"] + controls
    if weight_col is not None:
        needed.append(weight_col)
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    if weight_col is not None:
        sample = sample[sample[weight_col] > 0].copy()
    controls = keep_full_rank_controls(sample, controls)
    return sample.reset_index(drop=True), controls


def fit_iv(sample: pd.DataFrame, controls: list[str], weight_col: str | None) -> dict[str, float]:
    exog = pd.DataFrame({"const": 1.0}, index=sample.index)
    if controls:
        exog = pd.concat([exog, sample[controls]], axis=1)
    kwargs = {}
    if weight_col is not None:
        kwargs["weights"] = sample[weight_col]
    iv = IV2SLS(sample[Y], exog, sample[[X]], sample[[Z]], **kwargs).fit(cov_type="robust")

    fs_x = sm.add_constant(sample[[Z] + controls], has_constant="add")
    if weight_col is None:
        fs = sm.OLS(sample[X], fs_x).fit(cov_type="HC1")
    else:
        fs = sm.WLS(sample[X], fs_x, weights=sample[weight_col]).fit(cov_type="HC1")
    fstat = float(np.asarray(fs.f_test(f"{Z} = 0").fvalue).item())

    rf_x = sm.add_constant(sample[[Z] + controls], has_constant="add")
    if weight_col is None:
        rf = sm.OLS(sample[Y], rf_x).fit(cov_type="HC1")
    else:
        rf = sm.WLS(sample[Y], rf_x, weights=sample[weight_col]).fit(cov_type="HC1")

    return {
        "coef": float(iv.params[X]),
        "se": float(iv.std_errors[X]),
        "p": float(iv.pvalues[X]),
        "nobs": int(iv.nobs),
        "first_stage_coef": float(fs.params[Z]),
        "first_stage_F": fstat,
        "reduced_form_coef": float(rf.params[Z]),
    }


def bootstrap_iv(sample: pd.DataFrame, weight_col: str | None, rng: np.random.Generator) -> np.ndarray:
    states = sample["state"].dropna().unique()
    draws: list[float] = []
    for _ in range(BOOT_REPS):
        chosen = rng.choice(states, size=len(states), replace=True)
        boot = pd.concat(
            [sample[sample["state"] == state] for state in chosen],
            ignore_index=True,
        )
        try:
            controls = region_controls(boot)
            controls = keep_full_rank_controls(boot, controls)
            result = fit_iv(boot, controls, weight_col)
            if np.isfinite(result["coef"]):
                draws.append(result["coef"])
        except Exception:
            continue
    return np.asarray(draws, dtype=float)


def conley_interval(beta: float, first_stage: float, reduced_form: float, share: float) -> tuple[float, float]:
    """Return beta bounds allowing a direct instrument effect up to share*|RF|.

    With one excluded instrument, allowing direct effect gamma shifts the IV
    coefficient by -gamma / pi, where pi is the first-stage coefficient.
    """
    if not np.isfinite(first_stage) or abs(first_stage) < 1e-10:
        return np.nan, np.nan
    delta = share * abs(reduced_form)
    endpoints = [beta - gamma / first_stage for gamma in (-delta, delta)]
    return min(endpoints), max(endpoints)


def fmt_num(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def write_table(rows: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Soybean IV Bootstrap and Plausibly Exogenous Robustness}",
        r"\label{tab:soybean_bootstrap_conley_robustness}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{adjustbox}{max width=0.92\textwidth}",
        r"\begin{tabular}{lllc}",
        r"\hline\hline",
        r"Sample & Weighting & Statistic & Estimate or interval \\",
        r"\hline",
    ]
    for i, (_, row) in enumerate(rows.iterrows()):
        if i > 0:
            lines.append(r"\hline")
        sample = rf"Match score $>{int(row['threshold'])}\%$"
        weighting = row["weighting"]
        st = stars(float(row["p"]))
        coef = rf"${row['coef']:.3f}^{{{st}}}$" if st else rf"${row['coef']:.3f}$"
        boot_ci = rf"$[{row['boot_ci_low']:.3f}, {row['boot_ci_high']:.3f}]$"
        conley10 = rf"$[{row['conley_10_low']:.3f}, {row['conley_10_high']:.3f}]$"
        conley25 = rf"$[{row['conley_25_low']:.3f}, {row['conley_25_high']:.3f}]$"
        conley50 = rf"$[{row['conley_50_low']:.3f}, {row['conley_50_high']:.3f}]$"
        lines.extend(
            [
                rf"{sample} & {weighting} & IV estimate & {coef} \\",
                rf"{sample} & {weighting} & Bootstrap 95\% CI & {boot_ci} \\",
                rf"{sample} & {weighting} & First-stage F & ${row['first_stage_F']:.2f}$ \\",
                rf"{sample} & {weighting} & Conley bound, 10\% & {conley10} \\",
                rf"{sample} & {weighting} & Conley bound, 25\% & {conley25} \\",
                rf"{sample} & {weighting} & Conley bound, 50\% & {conley50} \\",
            ]
        )
    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.92\textwidth}",
        r"\footnotesize",
        rf"\emph{{Notes:}} Bootstrap intervals use {BOOT_REPS} state-resampling draws with replacement. All specifications include Census region fixed effects. The endogenous regressor is $\log(\hat{{\bar a}}_s)$ and the excluded instrument is log(1 + average historical soybean acres per 1,000 state land acres) over 1926--1940. Conley-style intervals allow the soybean instrument to have a direct reduced-form effect on log SNAP rates bounded by 10, 25, or 50 percent of the observed reduced-form soybean coefficient. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_bootstrap(draws_by_label: dict[str, np.ndarray], rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    point_estimates = []
    ci_lows = []
    ci_highs = []
    for label, draws in draws_by_label.items():
        if len(draws) == 0:
            continue
        weighting = label.split(",", 1)[0]
        row = rows[(rows["threshold"] == 95.0) & (rows["weighting"] == weighting)]
        if row.empty:
            continue
        row = row.iloc[0]
        labels.append(label)
        point_estimates.append(float(row["coef"]))
        ci_lows.append(float(row["boot_ci_low"]))
        ci_highs.append(float(row["boot_ci_high"]))

    y = np.arange(len(labels))
    lower_err = np.asarray(point_estimates) - np.asarray(ci_lows)
    upper_err = np.asarray(ci_highs) - np.asarray(point_estimates)
    ax.errorbar(
        point_estimates,
        y,
        xerr=[lower_err, upper_err],
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Soybean IV estimate")
    ax.set_title("Soybean IV Estimates with Bootstrap 95% Confidence Intervals")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run code/03_main_regressions.py first.")

    base = pd.read_csv(DATA)
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, float | str | int]] = []
    boot_plot: dict[str, np.ndarray] = {}

    for threshold in THRESHOLDS:
        for weighting, weight_col in WEIGHTS:
            sample, controls = estimation_sample(base, threshold, weight_col)
            result = fit_iv(sample, controls, weight_col)
            draws = bootstrap_iv(sample, weight_col, rng)
            if len(draws) < max(50, BOOT_REPS // 4):
                raise RuntimeError(
                    f"Too few successful bootstrap draws for threshold {threshold}, {weighting}: {len(draws)}"
                )
            ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
            row = {
                "threshold": threshold,
                "weighting": weighting,
                **result,
                "boot_reps_success": int(len(draws)),
                "boot_ci_low": float(ci_low),
                "boot_ci_high": float(ci_high),
            }
            for share in CONLEY_SHARES:
                low, high = conley_interval(
                    beta=result["coef"],
                    first_stage=result["first_stage_coef"],
                    reduced_form=result["reduced_form_coef"],
                    share=share,
                )
                pct = int(share * 100)
                row[f"conley_{pct}_low"] = low
                row[f"conley_{pct}_high"] = high
            rows.append(row)
            if threshold == 95.0:
                boot_plot[f"{weighting}, >95%"] = draws
            print(
                f"Finished threshold >{threshold:.0f}%, {weighting}: "
                f"coef={result['coef']:.3f}, bootstrap draws={len(draws)}"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "soybean_bootstrap_conley_robustness_results.csv", index=False)
    write_table(out, OUT / "soybean_bootstrap_conley_robustness_table.tex")
    plot_bootstrap(boot_plot, out, OUT / "soybean_bootstrap_distribution.png")
    print("Wrote soybean bootstrap and Conley robustness outputs.")


if __name__ == "__main__":
    main()
