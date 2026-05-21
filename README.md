# Cash-Crop Specialization, Subsistence Dependence, and SNAP Participation Rates Across U.S. States

## Replication Repository

This GitHub repository contains the public-facing replication materials for the paper:

> **Structural Transformation Beyond Productivity: Subsistence Organization and Agricultural Commercialization in the United States**

The repository is designed as a reproducibility archive for the project. It provides the analysis code, data-source documentation, and instructions needed to reconstruct the replication workflow.

At this stage, the repository is organized for GitHub rather than as a complete zipped replication package containing all raw and generated files.

---

## Current Repository Structure

```text
code/                 Python scripts for data cleaning, calibration, regressions, maps, and output checks.
data_directory.pdf    Separate data-directory guide explaining required raw files, filenames, sources, and download instructions.
README.md             Public repository overview and replication instructions.
.gitignore            Git ignore file for excluding temporary Python/system files.
```

Additional folders such as `raw_data/`, `clean_data/`, `derived/`, and `outputs/` are generated or reconstructed during replication and do not need to be fully stored in the GitHub repository.

---

## Replication Workflow

The main scripts should be run in the following order:

```text
01_clean_raw_data.py
→ 02_calibration_model.py
→ 03_main_regressions.py
→ 04_interaction_regressions.py
→ 05_check_outputs.py
→ 06_maps.py
```

The script `01_clean_raw_data.py` is the required first step. It reads the original source files from a local `raw_data/` directory and constructs the cleaned datasets used by the later scripts.

---

## Data Access and Raw-Data Documentation

Large raw data files are not necessarily distributed directly through this GitHub repository.

Instead, the file below provides detailed instructions for reconstructing the required raw-data directory:

```text
data_directory.pdf
```

This separate data-directory guide documents:

- required raw-data folders;
- original downloaded filenames;
- data-source links;
- IPUMS extract information;
- USDA ERS productivity and income files;
- USDA NASS Quick Stats crop and suitability downloads;
- USDA FNS SNAP participation data;
- U.S. Census population estimates;
- where each file should be placed before running the scripts.

---

## Original Data Sources

Researchers using this repository should cite the original data providers in addition to this replication repository.

### IPUMS USA

IPUMS USA. Integrated Public Use Microdata Series, Version 15.0. Minneapolis, MN: IPUMS.  
https://usa.ipums.org/usa/

### USDA Economic Research Service: Agricultural Productivity

United States Department of Agriculture, Economic Research Service.  
Agricultural Productivity in the United States.  
https://www.ers.usda.gov/data-products/agricultural-productivity-in-the-united-states

### USDA Economic Research Service: Farm Household Income

United States Department of Agriculture, Economic Research Service.  
Farm Household Income and Characteristics.  
https://www.ers.usda.gov/data-products/farm-household-income-and-characteristics

### USDA National Agricultural Statistics Service

United States Department of Agriculture, National Agricultural Statistics Service.  
Quick Stats Database.  
https://quickstats.nass.usda.gov/

### USDA Food and Nutrition Service

United States Department of Agriculture, Food and Nutrition Service.  
Supplemental Nutrition Assistance Program Participation and Costs.  
https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap

### U.S. Census Bureau

United States Census Bureau. Population Estimates Program.  
https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html

---

## Methodological Implementation Notes

The code implements the following methodological choices:

- Agricultural labor productivity is constructed from USDA ERS agricultural output and labor input quantity indices.
- The baseline calibration uses raw USDA ERS state labor productivity rather than a within-state 1960-normalized productivity series.
- State subsistence indices are calibrated using early-period moments from 1960 and 1970.
- Robustness calibration checks evaluate alternative early-year specifications.
- SNAP participation rates are constructed using USDA FNS SNAP participation counts and U.S. Census state population estimates.
- Historical crop instruments are based on planted acreage from USDA NASS Quick Stats.
- Fieldwork suitability is based on USDA NASS Quick Stats days-suitable-for-fieldwork data.

---

## Running the Code

Install the required Python packages:

```bash
pip install pandas numpy matplotlib openpyxl statsmodels linearmodels scipy tslearn geopandas mapclassify
```

Then run the scripts sequentially from the project root, following the order listed above.

If a `requirements.txt` file is added later, dependencies can instead be installed with:

```bash
pip install -r requirements.txt
```

---

## Generated Files

During replication, the scripts may create local folders such as:

```text
clean_data/
derived/
outputs/
```

These folders contain cleaned datasets, intermediate calibration outputs, regression results, figures, maps, and manuscript-facing tables. They can be regenerated from the code and source files described in `data_directory.pdf`.


