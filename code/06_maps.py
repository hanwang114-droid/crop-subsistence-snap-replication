# ============================================================
# INFORMATIVE STATE MAPS FOR SUBSISTENCE, SNAP, MATCHING,
# AND CROP SPECIALIZATION
#
# Replication-package version:
# - Uses package-relative directories only.
# - Finds calibration outputs robustly even if they are in derived/,
#   outputs/, output/, out/, calibration_output/, or clean_data/.
# - Uses the baseline raw-productivity calibration.
# - Creates maps for subsistence, SNAP, matching quality, and crop shares.
# ============================================================

# Anaconda Prompt:
# pip install pandas geopandas matplotlib mapclassify openpyxl

from pathlib import Path
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "raw_data"
CLEAN = ROOT / "clean_data"
DERIVED = ROOT / "derived"
OUTPUTS = ROOT / "outputs"

CALIB = DERIVED / "calibration"
MAP_OUT = OUTPUTS / "maps"
INTERACTION_OUT = DERIVED / "interaction"

MAP_OUT.mkdir(parents=True, exist_ok=True)
INTERACTION_OUT.mkdir(parents=True, exist_ok=True)

output_dir = MAP_OUT
data_path = INTERACTION_OUT / "final_dataset_weighted_interaction.csv"

# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 1926
END_YEAR = 1940

COTTON_VAR = "share_cotton"
SOY_VAR = "share_soybean"
CORN_VAR = "share_corn"

# ============================================================
# STATE MAPS
# ============================================================

state_to_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ",
    "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL",
    "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

abbr_to_region = {
    "AL": "South", "AR": "South", "DE": "South",
    "FL": "South", "GA": "South", "KY": "South",
    "LA": "South", "MD": "South", "MS": "South",
    "NC": "South", "OK": "South", "SC": "South",
    "TN": "South", "TX": "South", "VA": "South",
    "WV": "South",

    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest",
    "KS": "Midwest", "MI": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest",
    "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",

    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
    "NH": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast", "RI": "Northeast", "VT": "Northeast",

    "AK": "West", "AZ": "West", "CA": "West",
    "CO": "West", "HI": "West", "ID": "West",
    "MT": "West", "NV": "West", "NM": "West",
    "OR": "West", "UT": "West", "WA": "West",
    "WY": "West"
}

# ============================================================
# ROBUST FILE FINDERS
# ============================================================

def find_existing(explicit_candidates, recursive_patterns, required=True):
    """
    Find a file from explicit candidate paths first.
    If not found, search recursively under the package root.
    """

    for p in explicit_candidates:
        p = Path(p)
        if p.exists():
            return p

    matches = []
    for pattern in recursive_patterns:
        matches.extend(ROOT.rglob(pattern))

    matches = [
        p for p in matches
        if p.is_file()
        and "code" not in p.parts
        and "__pycache__" not in p.parts
    ]

    if matches:

        def score(p):
            name = p.name.lower()
            parent = str(p.parent).lower()
            s = 0
            if "raw_productivity" in name:
                s += 10
            if "baseline" in name or "1960_1970" in name:
                s += 8
            if "optimized_alpha" in name:
                s += 6
            if "calibration" in parent or "calib" in parent:
                s += 4
            if "common_alpha" in name:
                s -= 3
            return s

        matches = sorted(matches, key=score, reverse=True)
        return matches[0]

    if required:
        raise FileNotFoundError(
            "Could not find required file.\n\nTried explicit paths:\n"
            + "\n".join(str(p) for p in explicit_candidates)
            + "\n\nSearched recursively for:\n"
            + "\n".join(recursive_patterns)
        )

    return None


def find_crop_file(crop):
    """
    Find the raw QuickStats crop acreage file for one crop.
    Avoids the fieldwork-days-suitable file and other non-acreage files.
    """

    exact = [
        RAW / "quickstats" / f"{crop}.csv",
        RAW / "cash_crop_data" / f"{crop}.csv",
        CLEAN / "cash_crop_data" / f"{crop}.csv",
    ]

    for p in exact:
        if p.exists():
            return p

    patterns = {
        "soybean": ["*soybean*.csv", "*soybeans*.csv", "*soy*.csv"],
        "corn": ["*corn*.csv"],
        "cotton": ["*cotton*.csv"],
    }[crop]

    matches = []
    for pattern in patterns:
        matches.extend(ROOT.rglob(pattern))

    clean_matches = []
    for p in matches:
        lname = p.name.lower()
        if not p.is_file():
            continue
        if "code" in p.parts:
            continue
        if any(bad in lname for bad in ["suitable", "fieldwork", "days", "snap", "population"]):
            continue
        clean_matches.append(p)

    if not clean_matches:
        raise FileNotFoundError(f"Could not find crop file for {crop}.")

    def score(p):
        lname = p.name.lower()
        parent = str(p.parent).lower()
        s = 0
        if crop in lname:
            s += 10
        if "quickstats" in parent or "cash_crop" in parent:
            s += 5
        if "acres" in lname or "planted" in lname or "harvested" in lname:
            s += 3
        return s

    return sorted(clean_matches, key=score, reverse=True)[0]


# ============================================================
# INPUT FILES
# ============================================================

abar_path = find_existing(
    [
        DERIVED / "calibration" / "calibrated_state_abar_optimized_alpha_raw_productivity.csv",
        DERIVED / "calibration" / "calibrated_state_abar_raw_productivity_baseline_1960_1970.csv",
        OUTPUTS / "calibration" / "calibrated_state_abar_optimized_alpha_raw_productivity.csv",
        OUTPUTS / "calibration" / "calibrated_state_abar_raw_productivity_baseline_1960_1970.csv",
        ROOT / "calibration_output" / "calibrated_state_abar_optimized_alpha_raw_productivity.csv",
        CLEAN / "calibrated_state_abar_optimized_alpha_raw_productivity.csv",
    ],
    [
        "calibrated_state_abar_optimized_alpha_raw_productivity.csv",
        "calibrated_state_abar_raw_productivity_baseline_1960_1970.csv",
        "*abar*raw_productivity*baseline*1960*1970*.csv",
        "*abar*optimized_alpha*raw_productivity*.csv",
    ],
)

quality_path = find_existing(
    [
        DERIVED / "calibration" / "soft_dtw_matching_quality_by_state_raw_productivity.csv",
        DERIVED / "calibration" / "soft_dtw_matching_quality_raw_productivity_baseline_1960_1970.csv",
        OUTPUTS / "calibration" / "soft_dtw_matching_quality_by_state_raw_productivity.csv",
        OUTPUTS / "calibration" / "soft_dtw_matching_quality_raw_productivity_baseline_1960_1970.csv",
        ROOT / "calibration_output" / "soft_dtw_matching_quality_by_state_raw_productivity.csv",
        CLEAN / "soft_dtw_matching_quality_by_state_raw_productivity.csv",
    ],
    [
        "soft_dtw_matching_quality_by_state_raw_productivity.csv",
        "soft_dtw_matching_quality_raw_productivity_baseline_1960_1970.csv",
        "*soft_dtw*raw_productivity*baseline*1960*1970*.csv",
        "*matching_quality*raw_productivity*.csv",
    ],
)

snap_path = find_existing(
    [
        RAW / "snap" / "snap-persons-4.xlsx",
        CLEAN / "snap" / "snap-persons-4.xlsx",
    ],
    [
        "snap-persons-4.xlsx",
        "*snap*.xlsx",
    ],
)

pop_path = find_existing(
    [
        RAW / "population" / "NST-EST2025-POP.xlsx",
        CLEAN / "snap" / "NST-EST2025-POP.xlsx",
        RAW / "snap" / "NST-EST2025-POP.xlsx",
    ],
    [
        "NST-EST2025-POP.xlsx",
        "*POP*.xlsx",
        "*population*.xlsx",
    ],
)

crop_files = {
    "soybean": find_crop_file("soybean"),
    "corn": find_crop_file("corn"),
    "cotton": find_crop_file("cotton"),
}

clean_crop_path = CLEAN / "clean_crop_state_year.csv"

shapefile_path = find_existing(
    [
        RAW / "shapefiles" / "cb_2023_us_state_500k.zip",
        RAW / "shapefile" / "cb_2023_us_state_500k.zip",
        RAW / "maps" / "cb_2023_us_state_500k.zip",
    ],
    [
        "cb_2023_us_state_500k.zip",
        "*us_state_500k*.zip",
        "*state*500k*.zip",
    ],
    required=False
)

if shapefile_path is None:
    shapefile_path = (
        "https://www2.census.gov/geo/tiger/GENZ2023/shp/"
        "cb_2023_us_state_500k.zip"
    )

print("Using calibration file:", abar_path)
print("Using matching-quality file:", quality_path)
print("Using SNAP file:", snap_path)
print("Using population file:", pop_path)
print("Using shapefile:", shapefile_path)
print("Using crop files:")
for k, v in crop_files.items():
    print(f"  {k}: {v}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value_column(series):
    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(D)", "", regex=False)
        .str.replace("(Z)", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def clean_state_name(x):
    return str(x).replace(".", "").strip().title()


def state_series_to_abbr(series):
    s = series.astype(str).str.strip()
    return np.where(
        s.str.len() == 2,
        s.str.upper(),
        s.str.title().map(state_to_abbr)
    )


def truncate_colormap(cmap, minval=0.0, maxval=0.8, n=256):
    """
    Create a lighter version of a matplotlib colormap.
    This removes the darkest shades so publication maps look softer.
    """
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )


def make_informative_map(
    data,
    variable,
    title,
    subtitle,
    legend_title,
    filename,
    cmap="viridis",
    label_states=True
):
    plot_data = data.copy()

    if variable not in plot_data.columns:
        print(f"Skipping {filename}: variable not found -> {variable}")
        return

    cmap = truncate_colormap(
        plt.get_cmap(cmap),
        0.0,
        0.8
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    if variable == "matching_score_percent":

        plot_data.plot(
            column=variable,
            ax=ax,
            cmap=cmap,
            scheme="Quantiles",
            k=5,
            legend=True,
            edgecolor="black",
            linewidth=0.45,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "label": "Missing"
            },
            legend_kwds={
                "title": legend_title,
                "loc": "lower left",
                "fontsize": 9,
                "title_fontsize": 10,
                "frameon": True
            }
        )

    else:

        plot_data.plot(
            column=variable,
            ax=ax,
            cmap=cmap,
            scheme="Quantiles",
            k=5,
            legend=True,
            edgecolor="black",
            linewidth=0.45,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "black",
                "label": "Missing"
            },
            legend_kwds={
                "title": legend_title,
                "loc": "lower left",
                "fontsize": 9,
                "title_fontsize": 10,
                "frameon": True
            }
        )

    if label_states:
        for _, row in plot_data.dropna(subset=[variable]).iterrows():
            ax.text(
                row["centroid"].x,
                row["centroid"].y,
                row["state"],
                fontsize=7,
                ha="center",
                va="center",
                color="black"
            )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    if subtitle:
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.axis("off")

    n_states = plot_data[variable].notna().sum()

    note = (
        "Notes: Mainland U.S. states only; Alaska and Hawaii omitted for readability. "
        f"States grouped into five quantile bins. N = {n_states}."
    )

    ax.text(
        0.01,
        -0.03,
        note,
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="top"
    )

    plt.tight_layout()

    outpath = output_dir / filename

    plt.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", outpath)

# ============================================================
# 1. CREATE clean_crop_state_year.csv IF MISSING
# ============================================================

if not clean_crop_path.exists():

    print("clean_crop_state_year.csv not found. Creating it now...")

    cleaned = []

    for crop_name, path in crop_files.items():

        df_crop = pd.read_csv(path)

        if {"Geo Level", "Period", "Domain", "Value", "State", "Year"}.issubset(df_crop.columns):

            df_crop = df_crop[
                (df_crop["Geo Level"] == "STATE") &
                (df_crop["Period"] == "YEAR") &
                (df_crop["Domain"] == "TOTAL")
            ].copy()

            df_crop["value"] = clean_value_column(
                df_crop["Value"]
            )

            df_crop["state"] = (
                df_crop["State"]
                .astype(str)
                .str.title()
                .str.strip()
            )

            df_crop["year"] = pd.to_numeric(
                df_crop["Year"],
                errors="coerce"
            )

        else:

            cols = list(df_crop.columns)
            lower_map = {str(c).strip().lower(): c for c in cols}

            state_col = None
            for key in ["state", "state_name", "state name"]:
                if key in lower_map:
                    state_col = lower_map[key]
                    break

            year_col = None
            for key in ["year", "year_id"]:
                if key in lower_map:
                    year_col = lower_map[key]
                    break

            value_col = None
            for key in ["value", "acres", "acreage", "area planted", "area harvested"]:
                if key in lower_map:
                    value_col = lower_map[key]
                    break

            if value_col is None:
                numeric_candidates = []
                for col in cols:
                    temp = pd.to_numeric(
                        df_crop[col].astype(str).str.replace(",", "", regex=False),
                        errors="coerce"
                    )
                    if temp.notna().sum() > 10:
                        numeric_candidates.append(col)
                numeric_candidates = [c for c in numeric_candidates if c != year_col]
                if numeric_candidates:
                    value_col = numeric_candidates[-1]

            if state_col is None or year_col is None or value_col is None:
                raise ValueError(
                    f"Could not identify state/year/value columns in crop file: {path}\n"
                    f"Columns: {cols}"
                )

            df_crop["state"] = (
                df_crop[state_col]
                .astype(str)
                .str.title()
                .str.strip()
            )

            df_crop["year"] = pd.to_numeric(df_crop[year_col], errors="coerce")
            df_crop["value"] = clean_value_column(df_crop[value_col])

        df_crop["crop"] = crop_name

        df_crop = df_crop[
            ["state", "year", "crop", "value"]
        ]

        cleaned.append(df_crop)

    panel = pd.concat(
        cleaned,
        ignore_index=True
    )

    panel = panel.dropna(
        subset=["state", "year", "value"]
    )

    panel["year"] = panel["year"].astype(int)

    panel = panel.sort_values(
        ["state", "crop", "year"]
    )

    clean_crop_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_csv(
        clean_crop_path,
        index=False
    )

    print("Created:", clean_crop_path)

else:
    print("Found existing clean crop panel:", clean_crop_path)

# ============================================================
# 2. ALWAYS REBUILD final_dataset_weighted_interaction.csv
# ============================================================

print("Rebuilding final_dataset_weighted_interaction.csv...")

# ------------------------------
# LOAD ABAR
# ------------------------------

abar = pd.read_csv(abar_path)

abar["state"] = (
    abar["state"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ------------------------------
# LOAD MATCH QUALITY
# ------------------------------

quality = pd.read_csv(quality_path)

quality["state"] = (
    quality["state"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df_abar = abar.merge(
    quality[["state", "matching_score_percent"]],
    on="state",
    how="left"
)

df_abar["matching_score_percent"] = pd.to_numeric(
    df_abar["matching_score_percent"],
    errors="coerce"
)

df_abar["match_weight"] = (
    df_abar["matching_score_percent"] / 100
)

df_abar["match_weight_sq"] = (
    df_abar["match_weight"] ** 2
)

# ------------------------------
# LOAD SNAP
# ------------------------------

snap = pd.read_excel(
    snap_path,
    sheet_name=0,
    header=2
)

snap = snap.rename(
    columns={snap.columns[0]: "state_name"}
)

snap["state_name"] = (
    snap["state_name"]
    .astype(str)
    .str.title()
    .str.strip()
)

snap["state"] = (
    snap["state_name"]
    .map(state_to_abbr)
)

possible_cols = snap.columns[1:4]

snap["snap_persons"] = np.nan
snap_used_column = None

for col in possible_cols[::-1]:

    temp = pd.to_numeric(
        snap[col],
        errors="coerce"
    )

    if temp.notna().sum() > 20:

        snap["snap_persons"] = temp
        snap_used_column = col
        break

print("SNAP column used:", snap_used_column)

snap = snap.dropna(
    subset=["state", "snap_persons"]
).copy()

snap["snap_persons"] = pd.to_numeric(
    snap["snap_persons"],
    errors="coerce"
)

snap = snap.dropna(
    subset=["snap_persons"]
).copy()

snap["log_snap_persons"] = np.log(
    snap["snap_persons"]
)

snap = snap[
    [
        "state",
        "state_name",
        "snap_persons",
        "log_snap_persons"
    ]
].copy()

# ------------------------------
# LOAD CLEAN CROP PANEL
# ------------------------------

panel = pd.read_csv(clean_crop_path)

panel["state_name"] = (
    panel["state"]
    .astype(str)
    .str.title()
    .str.strip()
)

panel["state"] = (
    panel["state_name"]
    .map(state_to_abbr)
)

panel["year"] = pd.to_numeric(
    panel["year"],
    errors="coerce"
)

panel["value"] = pd.to_numeric(
    panel["value"],
    errors="coerce"
)

panel = panel.dropna(
    subset=["state", "year", "value"]
).copy()

panel["year"] = panel["year"].astype(int)

panel = panel[
    (panel["year"] >= START_YEAR) &
    (panel["year"] <= END_YEAR)
].copy()

# ------------------------------
# BUILD STATE-LEVEL CROP SHARES
# ------------------------------

crop_state = (
    panel
    .groupby(["state", "crop"], as_index=False)["value"]
    .mean()
)

crop_wide = (
    crop_state
    .pivot(index="state", columns="crop", values="value")
    .reset_index()
)

for crop in ["cotton", "soybean", "corn"]:

    if crop not in crop_wide.columns:
        crop_wide[crop] = 0.0

crop_wide[
    ["cotton", "soybean", "corn"]
] = crop_wide[
    ["cotton", "soybean", "corn"]
].fillna(0)

crop_wide["cash_crop_sum"] = (
    crop_wide["cotton"] +
    crop_wide["soybean"] +
    crop_wide["corn"]
)

crop_wide[COTTON_VAR] = np.where(
    crop_wide["cash_crop_sum"] > 0,
    crop_wide["cotton"] / crop_wide["cash_crop_sum"],
    np.nan
)

crop_wide[SOY_VAR] = np.where(
    crop_wide["cash_crop_sum"] > 0,
    crop_wide["soybean"] / crop_wide["cash_crop_sum"],
    np.nan
)

crop_wide[CORN_VAR] = np.where(
    crop_wide["cash_crop_sum"] > 0,
    crop_wide["corn"] / crop_wide["cash_crop_sum"],
    np.nan
)

crop_wide["z_cotton_over_cotton_corn_soy"] = crop_wide[COTTON_VAR]
crop_wide["z_soybean_over_cotton_corn_soy"] = crop_wide[SOY_VAR]
crop_wide["z_corn_over_cotton_corn_soy"] = crop_wide[CORN_VAR]

crop_wide["z_cotton_over_cotton_soy"] = np.where(
    (crop_wide["cotton"] + crop_wide["soybean"]) > 0,
    crop_wide["cotton"] / (
        crop_wide["cotton"] +
        crop_wide["soybean"]
    ),
    np.nan
)

# ------------------------------
# MERGE FINAL DATA
# ------------------------------

reg = (
    snap
    .merge(df_abar, on="state", how="inner")
    .merge(crop_wide, on="state", how="inner")
)

reg["region"] = (
    reg["state"]
    .map(abbr_to_region)
)

region_dummies = pd.get_dummies(
    reg["region"],
    prefix="region",
    drop_first=True,
    dtype=float
)

reg = pd.concat(
    [reg, region_dummies],
    axis=1
)

reg["abar_x_cotton"] = (
    reg["abar_state"] *
    reg[COTTON_VAR]
)

reg["high_cotton_median"] = (
    reg[COTTON_VAR] >= reg[COTTON_VAR].median()
).astype(float)

reg["abar_x_high_cotton"] = (
    reg["abar_state"] *
    reg["high_cotton_median"]
)

reg.to_csv(
    data_path,
    index=False
)

print("Created:", data_path)
print("Final dataset shape:", reg.shape)

# ============================================================
# 3. LOAD FINAL DATASET FOR MAPS
# ============================================================

df = pd.read_csv(data_path)

df["state"] = (
    df["state"]
    .astype(str)
    .str.upper()
    .str.strip()
)

print("Loaded final dataset:", data_path)
print("Dataset shape:", df.shape)

# ============================================================
# 4. LOAD SHAPEFILE
# ============================================================

states = gpd.read_file(shapefile_path)

states = states.rename(
    columns={"STUSPS": "state"}
)

states = states[
    ~states["state"].isin(
        ["PR", "GU", "VI", "MP", "AS"]
    )
].copy()

states_mainland = states[
    ~states["state"].isin(
        ["AK", "HI"]
    )
].copy()

map_df = states_mainland.merge(
    df,
    on="state",
    how="left"
)

map_df = map_df.to_crs("EPSG:5070")

map_df["centroid"] = (
    map_df.geometry.centroid
)

# ============================================================
# 5. MAKE MAPS
# ============================================================

make_informative_map(
    data=map_df,
    variable="abar_state",
    title="Calibrated Subsistence Parameter by State",
    subtitle="Darker colors indicate higher level of subsistence levels.",
    legend_title="Subsistence\nquintile",
    filename="informative_map_abar_state.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable="snap_persons",
    title="SNAP Participation by State",
    subtitle="",
    legend_title="SNAP persons\nquintile",
    filename="informative_map_snap_persons.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable="log_snap_persons",
    title="Log SNAP Participation by State",
    subtitle="",
    legend_title="Log SNAP\nquintile",
    filename="informative_map_log_snap_persons.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable="matching_score_percent",
    title="Calibration Matching Quality by State",
    subtitle="Darker colors indicate better Soft-DTW match between calibrated and observed trajectories.",
    legend_title="Match score",
    filename="informative_map_matching_score_percent.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable=COTTON_VAR,
    title="Historical Cotton Share by State",
    subtitle="",
    legend_title="Cotton share\nquintile",
    filename="informative_map_cotton_share.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable=SOY_VAR,
    title="Historical Soybean Share by State",
    subtitle="",
    legend_title="Soybean share\nquintile",
    filename="informative_map_soybean_share.png",
    cmap="Greys"
)

make_informative_map(
    data=map_df,
    variable=CORN_VAR,
    title="Historical Corn Share by State",
    subtitle="",
    legend_title="Corn share\nquintile",
    filename="informative_map_corn_share.png",
    cmap="Greys"
)

print("\nAll informative maps saved to:")
print(output_dir)
