
"""
Final replication-output check.

This script verifies that the main replication outputs exist after run_all.py.
It is intentionally simple and strict: if an expected table or figure folder is
missing, it raises an error so the replication package does not silently pass.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_CANDIDATES = [ROOT / "out", ROOT / "output", ROOT / "outputs"]
OUT = next((p for p in OUT_CANDIDATES if p.exists()), OUT_CANDIDATES[0])

# Calibration output folder may be either out/calibration or calibration_output,
# depending on the short-path package layout.
CALIB_CANDIDATES = [
    OUT / "calibration",
    OUT / "calib",
    ROOT / "calibration_output",
    OUT,
]
CALIB = next((p for p in CALIB_CANDIDATES if p.exists()), OUT)

required_patterns = [
    "**/*calibration*summary*.tex",
    "**/*early*robustness*.tex",
    "**/*average*data*model*.png",
    "**/county_level_regression_table.tex",
    "**/informative_map_county_snap_rate.png",
    "**/soybean_bootstrap_conley_robustness_table.tex",
    "**/soybean_bootstrap_distribution.png",
]

missing = []
for pattern in required_patterns:
    if not list(OUT.glob(pattern)) and not list(ROOT.glob(pattern)):
        missing.append(pattern)

state_dirs = list(ROOT.glob("**/state_model_fit_all"))
if not state_dirs:
    missing.append("state_model_fit_all/")

if state_dirs:
    state_dir = state_dirs[0]
    pngs = list(state_dir.glob("*.png"))
    if len(pngs) == 0:
        missing.append("state_model_fit_all/*.png")
    manifest = state_dir / "state_model_fit_manifest.csv"
    if not manifest.exists():
        missing.append("state_model_fit_all/state_model_fit_manifest.csv")

if missing:
    raise FileNotFoundError(
        "Missing required replication outputs:\n" + "\n".join(f" - {m}" for m in missing)
    )

print("All required calibration outputs are present, including state_model_fit_all/.")
