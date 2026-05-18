"""01_clean_raw_data.py

Build every clean analysis file directly from the original files in raw_data/.

This is the required first step in the replication workflow.  The rest of the
analysis scripts DO NOT read from raw_data/.  They read only the cleaned files
written by this script into clean_data/.

What this script does, in plain language
----------------------------------------
1. Reads the original IPUMS USA fixed-width extract and computes weighted
   agricultural/forestry/fishing labor-force shares by state and census year.
2. Reads USDA ERS state productivity accounts and constructs raw agricultural
   labor productivity as:
       Aa_labor_productivity_index =
       Total agricultural output quantity index / Labor input quantity index
   It does NOT normalize productivity within state to 1960 = 1.
3. Merges the IPUMS employment shares with the USDA productivity series to form
   the calibration panel used by the structural model.
4. Reads original USDA NASS Quick Stats downloads for corn, cotton, soybeans,
   and fieldwork days suitable.  It writes clean crop files and a clean
   suitability file used by the IV and interaction regressions.
5. Copies the original USDA ERS motivation files, SNAP file, and Census
   population file into clean_data/ after checking that they are the expected
   files.

Important method choices
------------------------
- Agricultural employment share is based on IPUMS IND1950 codes 105, 116, and
  126, corresponding to Agriculture, Forestry, and Fisheries.
- The denominator is the labor force.  The extract already applies the IPUMS
  case selection LABFORCE = 2, but the script checks/filter this again.
- Person weights PERWT are used when aggregating IPUMS microdata.
- Calibration productivity uses raw USDA ERS state labor productivity, not the
  old within-state 1960-normalized productivity series.
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
CLEAN = ROOT / "clean_data"

# Output folders used by the analysis scripts.
MOTIVATION_DIR = CLEAN / "motivation_data"
CROP_DIR = CLEAN / "cash_crop_data"
SNAP_DIR = CLEAN / "snap"
STRUCT_DIR = CLEAN / "structrual_transformation_data"  # spelling kept for compatibility with original scripts

for d in [MOTIVATION_DIR, CROP_DIR, SNAP_DIR, STRUCT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# State FIPS-to-abbreviation map used to aggregate IPUMS to the state level.
FIPS_TO_ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC",
    12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS",
    21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
    29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY",
    37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
    46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY"
}

AG_FOREST_FISH_IND1950 = {105, 116, 126}


def require(path: Path, label: str) -> None:
    """Stop with a clear message if a required raw file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def clean_number(series: pd.Series) -> pd.Series:
    """Convert Quick Stats values such as '1,200' or '(D)' to numeric."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(D)", "", regex=False)
        .str.replace("(Z)", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def copy_required(src: Path, dst: Path, label: str) -> None:
    """Copy a raw source file to clean_data after confirming it exists."""
    require(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_ipums_ag_share() -> pd.DataFrame:
    """Aggregate the raw IPUMS fixed-width file to state-year employment shares.

    The IPUMS .dat file is fixed-width.  The variable positions are documented in
    raw_data/ipums/code_book/usa_00001.xml and raw_data/ipums/code_book/usa_00001.txt.
    Only five columns are needed here: YEAR, STATEFIP, PERWT, LABFORCE, and IND1950.
    """
    dat = RAW / "ipums" / "data" / "usa_00001.dat"
    ddi = RAW / "ipums" / "code_book" / "usa_00001.xml"
    txt = RAW / "ipums" / "code_book" / "usa_00001.txt"
    require(dat, "IPUMS raw microdata file")
    require(ddi, "IPUMS DDI metadata file")
    require(txt, "IPUMS human-readable codebook")

    # Zero-based fixed-width column positions based on the IPUMS Stata/R code.
    colspecs = [(0, 4), (54, 56), (73, 83), (83, 84), (84, 87)]
    names = ["year", "statefip", "perwt", "labforce", "ind1950"]

    chunks = []
    for chunk in pd.read_fwf(dat, colspecs=colspecs, names=names, chunksize=1_000_000):
        # PERWT has two implied decimals in the IPUMS fixed-width extract.
        chunk["perwt"] = pd.to_numeric(chunk["perwt"], errors="coerce") / 100.0
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk["statefip"] = pd.to_numeric(chunk["statefip"], errors="coerce")
        chunk["labforce"] = pd.to_numeric(chunk["labforce"], errors="coerce")
        chunk["ind1950"] = pd.to_numeric(chunk["ind1950"], errors="coerce")

        # The extract was requested with LABFORCE = 2.  This line documents and
        # enforces that denominator explicitly.
        chunk = chunk[chunk["labforce"] == 2].copy()
        chunk = chunk[chunk["statefip"].isin(FIPS_TO_ABBR)].copy()
        chunk["state"] = chunk["statefip"].map(FIPS_TO_ABBR)
        chunk["ag_worker"] = chunk["ind1950"].isin(AG_FOREST_FISH_IND1950).astype(float)
        chunk["ag_weight"] = chunk["perwt"] * chunk["ag_worker"]

        agg = chunk.groupby(["state", "year"], as_index=False).agg(
            labor_force_weighted=("perwt", "sum"),
            ag_forest_fish_weighted=("ag_weight", "sum"),
        )
        chunks.append(agg)

    if not chunks:
        raise ValueError("No IPUMS observations were read. Check the .dat file.")

    out = pd.concat(chunks, ignore_index=True)
    out = out.groupby(["state", "year"], as_index=False).sum()
    out["ag_share_ag_forest_fish"] = out["ag_forest_fish_weighted"] / out["labor_force_weighted"]
    out = out.sort_values(["state", "year"])
    out.to_csv(STRUCT_DIR / "ipums_ag_employment_state_year.csv", index=False)
    return out


def build_productivity_panel() -> pd.DataFrame:
    """Convert USDA ERS state productivity accounts from long to state-year wide format."""
    src = RAW / "usda" / "Table02StateProductivityAccounts.csv"
    require(src, "USDA ERS state productivity accounts")
    df = pd.read_csv(src)
    required = {"State", "Year", "Variable Name", "Value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"USDA productivity file is missing columns: {sorted(missing)}")

    keep_vars = [
        "Total agricultural output quantity index",
        "Labor input quantity index",
        "Total factor productivity",
    ]
    df = df[df["Variable Name"].isin(keep_vars)].copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    wide = df.pivot_table(index=["State", "Year"], columns="Variable Name", values="Value", aggfunc="first").reset_index()
    wide = wide.rename(columns={"State": "state", "Year": "year"})
    wide["Aa_labor_productivity_index"] = (
        wide["Total agricultural output quantity index"] / wide["Labor input quantity index"]
    )
    wide["TFP"] = wide.get("Total factor productivity")

    # A normalized column is provided only as a diagnostic.  The calibration
    # script never uses this column.
    wide["Aa_labor_productivity_index_1960_state1"] = np.nan
    for state, g in wide.groupby("state"):
        base = g.loc[g["year"] == 1960, "Aa_labor_productivity_index"]
        if not base.empty and pd.notna(base.iloc[0]) and base.iloc[0] != 0:
            wide.loc[wide["state"] == state, "Aa_labor_productivity_index_1960_state1"] = (
                wide.loc[wide["state"] == state, "Aa_labor_productivity_index"] / base.iloc[0]
            )
    return wide


def build_calibration_panel() -> None:
    """Create the final state-year panel used by the structural calibration."""
    ipums = build_ipums_ag_share()
    prod = build_productivity_panel()
    panel = ipums.merge(prod, on=["state", "year"], how="inner")
    panel = panel.dropna(subset=["ag_share_ag_forest_fish", "Aa_labor_productivity_index"])
    panel = panel.sort_values(["state", "year"])
    panel.to_csv(STRUCT_DIR / "calibration_state_year_panel.csv", index=False)
    print(f"Wrote calibration panel with {len(panel):,} state-year observations.")


def identify_quickstats_files() -> dict:
    """Identify which raw Quick Stats CSV corresponds to each required variable."""
    files = list((RAW / "quickstats").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No Quick Stats CSV files found in raw_data/quickstats/.")
    found = {}
    for path in files:
        head = pd.read_csv(path, nrows=50)
        commodities = set(head.get("Commodity", pd.Series(dtype=str)).dropna().astype(str).str.upper())
        data_items = " | ".join(head.get("Data Item", pd.Series(dtype=str)).dropna().astype(str).str.upper().unique())
        if "SOYBEANS" in commodities and "ACRES PLANTED" in data_items:
            found["soybean"] = path
        elif "CORN" in commodities and "ACRES PLANTED" in data_items:
            found["corn"] = path
        elif "COTTON" in commodities and "ACRES PLANTED" in data_items:
            found["cotton"] = path
        elif "FIELDWORK" in commodities and "DAYS SUITABLE" in data_items:
            found["suitability"] = path
    missing = {"soybean", "corn", "cotton", "suitability"}.difference(found)
    if missing:
        raise ValueError(f"Could not identify these Quick Stats raw files: {sorted(missing)}")
    return found


def clean_crop_file(src: Path, crop: str) -> None:
    """Write a clean state-year crop acreage file from an original Quick Stats CSV."""
    df = pd.read_csv(src)
    df = df[
        (df["Geo Level"].astype(str).str.upper() == "STATE")
        & (df["Domain"].astype(str).str.upper() == "TOTAL")
        & (df["Data Item"].astype(str).str.upper().str.contains("ACRES PLANTED", na=False))
    ].copy()
    df["value"] = clean_number(df["Value"])
    df["State"] = df["State"].astype(str).str.upper().str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["State", "Year", "value"])
    out = df.rename(columns={"State": "State", "Year": "Year", "Geo Level": "Geo Level", "Period": "Period", "Domain": "Domain", "Value": "Value"})
    # Keep original Quick Stats column names expected by the analysis scripts.
    out.to_csv(CROP_DIR / f"{crop}.csv", index=False)
    print(f"Wrote clean crop file: {crop}.csv")


def clean_suitability_file(src: Path) -> None:
    """Clean Quick Stats fieldwork days suitable into state-week suitability data."""
    df = pd.read_csv(src)
    df = df[
        (df["Geo Level"].astype(str).str.upper() == "STATE")
        & (df["Domain"].astype(str).str.upper() == "TOTAL")
        & (df["Data Item"].astype(str).str.upper().str.contains("DAYS SUITABLE", na=False))
    ].copy()
    df["suitability"] = clean_number(df["Value"])
    df["state"] = df["State"].astype(str).str.upper().str.strip()
    df["week_ending"] = pd.to_datetime(df["Week Ending"], errors="coerce")
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["state", "week_ending", "suitability"])
    df[["state", "week_ending", "year", "suitability"]].to_csv(CROP_DIR / "suitability_clean.csv", index=False)
    print("Wrote clean suitability file: suitability_clean.csv")


def build_quickstats_clean_files() -> None:
    """Clean crop acreage and fieldwork suitability files from raw Quick Stats exports."""
    found = identify_quickstats_files()
    for crop in ["soybean", "corn", "cotton"]:
        clean_crop_file(found[crop], crop)
    clean_suitability_file(found["suitability"])


def copy_other_sources() -> None:
    """Copy source files that analysis scripts read directly after validation."""
    copy_required(RAW / "usda" / "table01.xlsx", MOTIVATION_DIR / "table01.xlsx", "USDA ERS national productivity workbook")
    copy_required(RAW / "income" / "household-income-2026-02.csv", MOTIVATION_DIR / "household-income-2026-02.csv", "USDA ERS farm household income CSV")
    copy_required(RAW / "snap" / "snap-persons-5.xlsx", SNAP_DIR / "snap-persons-4.xlsx", "USDA FNS SNAP persons workbook")
    copy_required(RAW / "population" / "NST-EST2025-POP.xlsx", SNAP_DIR / "NST-EST2025-POP.xlsx", "Census population workbook")


def main() -> None:
    print("Cleaning original source files from raw_data/ into clean_data/ ...")
    copy_other_sources()
    build_quickstats_clean_files()
    build_calibration_panel()
    print("Done. All analysis scripts will now read from clean_data/ only.")


if __name__ == "__main__":
    main()
