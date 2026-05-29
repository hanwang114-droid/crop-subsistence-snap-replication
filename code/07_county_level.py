from __future__ import annotations

import csv
import math
import json
import os
import getpass
import zipfile
from itertools import combinations
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
NHGIS = RAW / "nhgis"
GAZETTEER_RAW = RAW / "census_gazetteer"
OUT = ROOT / "outputs"
CLEAN = ROOT / "clean_data"
TABLES = OUT / "county"

ACS_YEAR = "2024"
ACS_DATASET = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip"
SQMETERS_PER_ACRE = 4046.8564224

ACS_VARS = {
    "NAME": "name",
    "B22001_001E": "hh_total",
    "B22001_002E": "hh_snap",
    "B01003_001E": "population",
    "B19013_001E": "median_hh_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_count",
    "B23025_003E": "civilian_labor_force",
    "B23025_005E": "unemployed",
    "B02001_003E": "black_population",
    "B03003_003E": "hispanic_population",
    "B15003_001E": "educ25_total",
    "B15003_022E": "ba",
    "B15003_023E": "masters",
    "B15003_024E": "professional",
    "B15003_025E": "doctorate",
}

CONTROL_GROUPS = {
    "Population": ["log_population"],
    "Demographics": ["black_share", "hispanic_share", "ba_plus_share"],
    "Socioeconomic": ["log_median_hh_income", "poverty_rate", "unemployment_rate"],
}


def fetch_json(url: str) -> list[list[str]]:
    with urlopen(url, timeout=90) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API did not return JSON: {body[:1000]}") from exc


def download_acs() -> Path:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        key = getpass.getpass("Enter Census API key: ").strip()
        if not key:
            raise RuntimeError("A Census API key is required to download ACS data.")
        os.environ["CENSUS_API_KEY"] = key

    params = {
        "get": ",".join(ACS_VARS.keys()),
        "for": "county:*",
        "in": "state:*",
        "key": key,
    }
    rows = fetch_json(f"{ACS_DATASET}?{urlencode(params)}")
    header = rows[0]
    data = rows[1:]
    out_header = [ACS_VARS.get(col, col) for col in header] + ["county_fips_full"]
    state_idx = header.index("state")
    county_idx = header.index("county")

    CLEAN.mkdir(parents=True, exist_ok=True)
    out = CLEAN / f"acs_county_{ACS_YEAR}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for row in data:
            state = row[state_idx].zfill(2)
            county = row[county_idx].zfill(3)
            writer.writerow(row + [state + county])

    print(f"Wrote {out}")
    return out


def download_county_land_area() -> Path:
    CLEAN.mkdir(parents=True, exist_ok=True)
    GAZETTEER_RAW.mkdir(parents=True, exist_ok=True)
    raw_zip = GAZETTEER_RAW / "census_2024_county_gazetteer.zip"
    out = CLEAN / "county_land_area_2024.csv"

    with urlopen(GAZETTEER_URL, timeout=90) as response:
        raw_zip.write_bytes(response.read())

    with zipfile.ZipFile(raw_zip) as zf:
        txt_names = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        if not txt_names:
            raise FileNotFoundError("No text file found in Census county gazetteer zip.")
        gaz = pd.read_csv(zf.open(txt_names[0]), sep="\t", dtype=str)

    gaz.columns = [col.strip() for col in gaz.columns]
    geoid_col = "GEOID" if "GEOID" in gaz.columns else "GEOIDFP"
    aland_col = "ALAND" if "ALAND" in gaz.columns else "ALAND_SQMI"
    land = pd.DataFrame({"county_fips_full": gaz[geoid_col].str.zfill(5)})
    if aland_col == "ALAND":
        land["land_acres"] = pd.to_numeric(gaz[aland_col], errors="coerce") / SQMETERS_PER_ACRE
    else:
        land["land_acres"] = pd.to_numeric(gaz[aland_col], errors="coerce") * 640
    land["log_land_acres"] = np.nan
    land.loc[land["land_acres"] > 0, "log_land_acres"] = np.log(land.loc[land["land_acres"] > 0, "land_acres"])
    land.to_csv(out, index=False)
    print(f"Wrote {out}")
    return out


def fips_from_gisjoin(gisjoin: pd.Series) -> pd.Series:
    text = gisjoin.astype(str)
    state = text.str.slice(1, 3)
    county = text.str.slice(4, 7)
    return state + county


def clean_nhgis_soybean() -> Path:
    csv_files = sorted(NHGIS.glob("*_1920_county.csv"))
    if not csv_files:
        csv_files = sorted(NHGIS.glob("*county*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No NHGIS county CSV found in {NHGIS}")

    df = pd.read_csv(csv_files[0], dtype={"GISJOIN": str})
    if "AB4X001" not in df.columns:
        raise ValueError("Expected NHGIS column AB4X001 for soybean acreage, 1919.")

    out_df = pd.DataFrame(
        {
            "county_fips_full": fips_from_gisjoin(df["GISJOIN"]),
            "state_name_historical": df["STATE"],
            "county_name_historical": df["COUNTY"],
            "soybean_acres_1919": pd.to_numeric(df["AB4X001"], errors="coerce").fillna(0),
        }
    )
    out_df["log1p_soybean_acres_1919"] = np.log1p(out_df["soybean_acres_1919"])

    CLEAN.mkdir(parents=True, exist_ok=True)
    out = CLEAN / "nhgis_soybean_1919.csv"
    out_df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_acs_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "hh_total",
        "hh_snap",
        "population",
        "median_hh_income",
        "poverty_universe",
        "poverty_count",
        "civilian_labor_force",
        "unemployed",
        "black_population",
        "hispanic_population",
        "educ25_total",
        "ba",
        "masters",
        "professional",
        "doctorate",
    ]:
        df[col] = numeric(df[col])

    df["snap_rate"] = df["hh_snap"] / df["hh_total"]
    df["poverty_rate"] = df["poverty_count"] / df["poverty_universe"]
    df["unemployment_rate"] = df["unemployed"] / df["civilian_labor_force"]
    df["black_share"] = df["black_population"] / df["population"]
    df["hispanic_share"] = df["hispanic_population"] / df["population"]
    df["ba_plus"] = df[["ba", "masters", "professional", "doctorate"]].sum(axis=1)
    df["ba_plus_share"] = df["ba_plus"] / df["educ25_total"]
    df["log_population"] = np.nan
    df.loc[df["population"] > 0, "log_population"] = np.log(df.loc[df["population"] > 0, "population"])
    df["log_median_hh_income"] = np.nan
    df.loc[df["median_hh_income"] > 0, "log_median_hh_income"] = np.log(
        df.loc[df["median_hh_income"] > 0, "median_hh_income"]
    )
    df["log_snap_rate"] = np.nan
    df.loc[df["snap_rate"] > 0, "log_snap_rate"] = np.log(df.loc[df["snap_rate"] > 0, "snap_rate"])
    return df


def make_panel(acs_path: Path, soy_path: Path, land_path: Path) -> pd.DataFrame:
    acs = pd.read_csv(acs_path, dtype={"state": str, "county": str, "county_fips_full": str})
    soy = pd.read_csv(soy_path, dtype={"county_fips_full": str})
    land = pd.read_csv(land_path, dtype={"county_fips_full": str})
    acs["county_fips_full"] = acs["county_fips_full"].str.zfill(5)
    soy["county_fips_full"] = soy["county_fips_full"].str.zfill(5)
    land["county_fips_full"] = land["county_fips_full"].str.zfill(5)
    panel = add_acs_variables(acs).merge(soy, on="county_fips_full", how="inner")
    panel = panel.merge(land, on="county_fips_full", how="left")
    panel["soybean_acres_per_1000_land_acres"] = np.nan
    valid_land = panel["land_acres"] > 0
    panel.loc[valid_land, "soybean_acres_per_1000_land_acres"] = (
        1000 * panel.loc[valid_land, "soybean_acres_1919"] / panel.loc[valid_land, "land_acres"]
    )
    panel["log1p_soybean_per_1000_land_acres"] = np.nan
    valid_soy_intensity = panel["soybean_acres_per_1000_land_acres"] >= 0
    panel.loc[valid_soy_intensity, "log1p_soybean_per_1000_land_acres"] = np.log1p(
        panel.loc[valid_soy_intensity, "soybean_acres_per_1000_land_acres"]
    )
    out = CLEAN / "county_historical_soybean_panel.csv"
    panel.to_csv(out, index=False)
    print(f"Wrote {out}")
    return panel


def control_specs() -> list[tuple[str, list[str]]]:
    names = list(CONTROL_GROUPS)
    specs: list[tuple[str, list[str]]] = [("No controls", [])]
    for size in [1, 2, 3]:
        for combo in combinations(names, size):
            label = " + ".join(combo)
            controls = [col for group in combo for col in CONTROL_GROUPS[group]]
            specs.append((label, controls))
    return specs


def design_matrix(df: pd.DataFrame, treatment: str, controls: list[str]) -> tuple[np.ndarray, list[str]]:
    parts = [pd.Series(1.0, index=df.index, name="const"), df[[treatment] + controls]]
    state_fe = pd.get_dummies(df["state"], prefix="state", drop_first=True, dtype=float)
    parts.append(state_fe)
    xdf = pd.concat(parts, axis=1)
    return xdf.to_numpy(dtype=float), list(xdf.columns)


def ols_cluster(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        idx = clusters == cluster
        score = x[idx, :].T @ resid[idx]
        meat += np.outer(score, score)
    n, k = x.shape
    g = len(unique_clusters)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    vcov = correction * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(vcov))


def run_model(
    df: pd.DataFrame,
    outcome: str,
    outcome_label: str,
    controls_label: str,
    controls: list[str],
) -> dict[str, float | str | int]:
    treatment = "log1p_soybean_per_1000_land_acres"
    cols = [outcome, treatment, "state"] + controls
    sample = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = sample[outcome].to_numpy(dtype=float)
    x, names = design_matrix(sample, treatment, controls)
    beta, se = ols_cluster(y, x, sample["state"].to_numpy())
    j = names.index(treatment)
    t_stat = beta[j] / se[j]
    return {
        "outcome": outcome_label,
        "treatment": "Log soybean acres per 1,000 land acres, 1919",
        "control_groups": controls_label,
        "coef": beta[j],
        "cluster_se_state": se[j],
        "t_stat": t_stat,
        "p_value": math.erfc(abs(t_stat) / math.sqrt(2)),
        "n": len(sample),
        "state_clusters": sample["state"].nunique(),
        "controls": ", ".join(controls),
    }


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "^{***}"
    if p_value < 0.05:
        return "^{**}"
    if p_value < 0.10:
        return "^{*}"
    return ""


def format_coef(coef: float, p_value: float) -> str:
    return f"${coef:.3f}{significance_stars(p_value)}$"


def format_se(se: float) -> str:
    return f"$({se:.3f})$"


def has_control(control_groups: str, control_name: str) -> str:
    groups = [part.strip() for part in control_groups.split("+")]
    return "Yes" if control_name in groups else "No"


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    model_numbers = [f"({i})" for i in range(1, len(df) + 1)]
    coef_cells = [format_coef(row.coef, row.p_value) for row in df.itertuples()]
    se_cells = [format_se(row.cluster_se_state) for row in df.itertuples()]
    population_controls = [has_control(row.control_groups, "Population") for row in df.itertuples()]
    demographic_controls = [has_control(row.control_groups, "Demographics") for row in df.itertuples()]
    socioeconomic_controls = [has_control(row.control_groups, "Socioeconomic") for row in df.itertuples()]
    state_fe = ["Yes"] * len(df)
    observations = [f"{int(row.n):,}" for row in df.itertuples()]
    states = [f"{int(row.state_clusters)}" for row in df.itertuples()]

    col_spec = "l" + "c" * len(df)

    def row(label: str, cells: list[str]) -> str:
        return f"{label} & " + " & ".join(cells) + " \\\\"

    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{County-Level Reduced-Form Estimates: Historical Soybean Intensity and SNAP Receipt}",
        "\\label{tab:county_reduced_form}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline\\hline",
        row("", model_numbers),
        "\\hline",
        row("Log soybean acres per 1,000 land acres, 1919", coef_cells),
        row("", se_cells),
        row("Population controls", population_controls),
        row("Demographic controls", demographic_controls),
        row("Socioeconomic controls", socioeconomic_controls),
        row("State fixed effects", state_fe),
        row("Observations", observations),
        row("States", states),
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\vspace{0.25em}",
        "\\begin{minipage}{0.98\\textwidth}",
        "\\footnotesize",
        "\\emph{Notes:} The dependent variable is the log county SNAP household receipt rate from the ACS 2024 five-year data. "
        "The treatment is log one plus 1919 soybean acres per 1,000 county land acres, constructed from IPUMS NHGIS county agricultural tables and Census Gazetteer county land area. "
        "Population controls include log population. Demographic controls include Black share, Hispanic share, and BA-plus share. "
        "Socioeconomic controls include log median household income, poverty rate, and unemployment rate. "
        "All specifications include state fixed effects. State-clustered standard errors are reported in parentheses. "
        "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_regressions(panel: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    snap_results = [
        run_model(panel, "log_snap_rate", "Log SNAP household receipt rate", label, controls)
        for label, controls in control_specs()
    ]
    snap_df = pd.DataFrame(snap_results)
    snap_tex = TABLES / "county_level_regression_table.tex"
    write_latex_table(snap_df, snap_tex)
    print(f"Wrote {snap_tex}")


def main() -> None:
    acs_path = download_acs()
    land_path = download_county_land_area()
    soy_path = clean_nhgis_soybean()
    panel = make_panel(acs_path, soy_path, land_path)
    run_regressions(panel)
    print("Done.")


if __name__ == "__main__":
    main()
