# -----------------------------------------------------------------------------
# This script is part of the replication workflow. It assumes that
# code/00_clean_data.py has already converted original files in raw_data/ into
# clean_data/. This script reads cleaned inputs only; it does not read raw_data/.
# -----------------------------------------------------------------------------

"""Create the motivation figures and regression table used in the paper.

This script combines two national USDA--ERS series:
  * agricultural total factor productivity (TFP), and
  * the farm-household-income ratio.

It estimates two descriptive OLS models:
  1. TFP on the farm income ratio;
  2. the same model with a post-2000 break and an interaction.

It writes exactly three manuscript-facing files to outputs/motivation/:
  * tfp_farm_ratio_timeseries.png
  * ols_break_regression.png
  * regression_table.tex
"""
# -----------------------------------------------------------------------------
# READER GUIDE
# -----------------------------------------------------------------------------
# This file creates the motivation figure and table only. It does not affect the
# main state-level regressions. The code reads two national USDA ERS files,
# merges them by year, estimates the two regressions described in the appendix,
# and writes the exact LaTeX table and figures used in the paper.
# -----------------------------------------------------------------------------

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "clean_data" / "motivation_data"
OUT_DIR = ROOT / "outputs" / "motivation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFP_FILE = DATA_DIR / "table01.xlsx"
INCOME_FILE = DATA_DIR / "household-income-2026-02.csv"
BREAK_YEAR = 2000


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef(coef, p):
    st = stars(p)
    return f"${coef:.3f}^{{{st}}}$" if st else f"${coef:.3f}$"


def fmt_se(se):
    return f"$({se:.3f})$"


def load_data():
    tfp = pd.read_excel(TFP_FILE, sheet_name="Table 1", header=2)
    tfp = tfp[["Year", "Total factor productivity (TFP)"]].copy()
    tfp.columns = ["year", "tfp"]
    tfp["year"] = pd.to_numeric(tfp["year"], errors="coerce")
    tfp["tfp"] = pd.to_numeric(tfp["tfp"], errors="coerce")
    tfp = tfp[(tfp["year"] >= 1960) & (tfp["year"] <= 2024)].dropna()

    income = pd.read_csv(INCOME_FILE)
    income = income[
        income["IncomeMeasure"] == "Farm household: Ratio of farm to total income (ratio)"
    ][["Year", "Value"]].copy()
    income.columns = ["year", "farm_ratio"]
    income["year"] = pd.to_numeric(income["year"], errors="coerce")
    income["farm_ratio"] = pd.to_numeric(income["farm_ratio"], errors="coerce")
    income = income[(income["year"] >= 1960) & (income["year"] <= 2024)].dropna()

    df = pd.merge(tfp, income, on="year", how="inner").sort_values("year").reset_index(drop=True)
    df["post2000"] = (df["year"] >= BREAK_YEAR).astype(int)
    df["interaction"] = df["farm_ratio"] * df["post2000"]
    return df


def estimate(df):
    y = df["tfp"]
    m1 = sm.OLS(y, sm.add_constant(df[["farm_ratio"]])).fit()
    m2 = sm.OLS(y, sm.add_constant(df[["farm_ratio", "post2000", "interaction"]])).fit()

    cov = m2.cov_params()
    implied_coef = m2.params["farm_ratio"] + m2.params["interaction"]
    implied_se = math.sqrt(
        cov.loc["farm_ratio", "farm_ratio"]
        + cov.loc["interaction", "interaction"]
        + 2 * cov.loc["farm_ratio", "interaction"]
    )
    implied_p = float(m2.t_test("farm_ratio + interaction = 0").pvalue)
    return m1, m2, {"coef": implied_coef, "se": implied_se, "p": implied_p}


def write_latex_table(m1, m2, implied):
    latex = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Motivating Relationship Between Agricultural Productivity and Farm Income Share}}
\label{{tab:regression_results}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{adjustbox}}{{max width=0.98\textwidth}}
\begin{{tabular}}{{lccccccc}}
\hline\hline
& Farm income & Post-2000 & Farm income $\times$ & Implied post-2000 & Constant & $R^2$ & Obs. \\
& ratio & dummy & Post-2000 & slope &  &  &  \\
\hline
OLS & {fmt_coef(m1.params['farm_ratio'], m1.pvalues['farm_ratio'])} & -- & -- & -- & {fmt_coef(m1.params['const'], m1.pvalues['const'])} & {m1.rsquared:.3f} & {int(m1.nobs)} \\
& {fmt_se(m1.bse['farm_ratio'])} &  &  &  & {fmt_se(m1.bse['const'])} &  &  \\
OLS + break & {fmt_coef(m2.params['farm_ratio'], m2.pvalues['farm_ratio'])} & {fmt_coef(m2.params['post2000'], m2.pvalues['post2000'])} & {fmt_coef(m2.params['interaction'], m2.pvalues['interaction'])} & {fmt_coef(implied['coef'], implied['p'])} & {fmt_coef(m2.params['const'], m2.pvalues['const'])} & {m2.rsquared:.3f} & {int(m2.nobs)} \\
& {fmt_se(m2.bse['farm_ratio'])} & {fmt_se(m2.bse['post2000'])} & {fmt_se(m2.bse['interaction'])} & {fmt_se(implied['se'])} & {fmt_se(m2.bse['const'])} &  &  \\
\hline\hline
\end{{tabular}}
\end{{adjustbox}}
\vspace{{0.25em}}
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize
\emph{{Notes:}} The dependent variable is Total Factor Productivity (TFP). 
The farm income ratio is the ratio of farm household income to total household income. 
The post-2000 dummy equals one for years 2000 and after. 
The implied post-2000 slope is the sum of the farm income ratio coefficient and the interaction coefficient in the break specification. 
Standard errors are in parentheses. 
$^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
\end{{minipage}}
\end{{table}}
"""
    (OUT_DIR / "regression_table.tex").write_text(latex, encoding="utf-8")


def make_graphs(df, m2):
    b0 = m2.params["const"]
    b1 = m2.params["farm_ratio"]
    b2 = m2.params["post2000"]
    b3 = m2.params["interaction"]

    df_pre = df[df["year"] < BREAK_YEAR].copy().sort_values("farm_ratio")
    df_post = df[df["year"] >= BREAK_YEAR].copy().sort_values("farm_ratio")

    y_pre_hat = b0 + b1 * df_pre["farm_ratio"]
    y_post_hat = b0 + b2 + (b1 + b3) * df_post["farm_ratio"]

    plt.figure(figsize=(9, 6))
    plt.scatter(df_pre["farm_ratio"], df_pre["tfp"], alpha=0.8, label=f"Before {BREAK_YEAR}")
    plt.scatter(df_post["farm_ratio"], df_post["tfp"], alpha=0.8, label=f"{BREAK_YEAR} and after")
    plt.plot(df_pre["farm_ratio"], y_pre_hat, linewidth=2)
    plt.plot(df_post["farm_ratio"], y_post_hat, linewidth=2)
    plt.xlabel("Farm income / total income ratio")
    plt.ylabel("Total factor productivity (TFP)")
    plt.title("OLS with Post-2000 Break")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ols_break_regression.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["year"], df["tfp"], linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total factor productivity (TFP)")
    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["farm_ratio"], linestyle="--", linewidth=2)
    ax2.set_ylabel("Farm income ratio")
    ax1.axvline(x=BREAK_YEAR, linestyle=":", linewidth=2)
    plt.title("Time Series of TFP and Farm Income Ratio")
    fig.tight_layout()
    plt.savefig(OUT_DIR / "tfp_farm_ratio_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()
    m1, m2, implied = estimate(df)
    write_latex_table(m1, m2, implied)
    make_graphs(df, m2)
    print("Motivation outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
