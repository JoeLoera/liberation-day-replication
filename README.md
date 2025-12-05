# Liberation Day Tariffs: Python Replication Package

**Python conversion of "Making America Great Again? The Economic Impacts of Liberation Day Tariffs"**

By Ignatenko, Macedoni, Lashkaripour, and Simonovska (2025)

[![Status](https://img.shields.io/badge/status-67%25%20complete-yellow)](https://github.com/JoeLoera/liberation-day-replication)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

## Overview

This repository contains a Python conversion of the MATLAB replication package for analyzing the economic impacts of proposed "Liberation Day" tariffs. The conversion is **67% complete** with 6 out of 9 tables successfully replicated.

### Paper Citation

> Ignatenko, A., Macedoni, L., Lashkaripour, A., & Simonovska, I. (2025).
> Making America Great Again? The Economic Impacts of Liberation Day Tariffs.

## Project Status

### ✅ Working (6/9 Tables)

| Table | Description | Status |
|-------|-------------|--------|
| Table 1 | Baseline policy scenarios | ✅ Perfect match |
| Table 2 | Retaliation scenarios | ✅ Perfect match |
| Table 3 | Tariff revenue | ✅ Perfect match |
| Table 8 | Regional trade wars | ✅ Perfect match |
| Table 9 | Alternative specifications (Eaton-Kortum) | ✅ Perfect match |
| Table 10 | Deficit frameworks | ✅ Partial match (2/4 cases) |

### ⏸️ In Progress (3/9 Tables)

| Table | Description | Status |
|-------|-------------|--------|
| Table 4 | IO model with one sector | ⏸️ Convergence issues |
| Table 7 | Trade elasticity estimation | ⏸️ Convergence issues |
| Table 11 | IO model alternative specifications | ⏸️ Convergence issues |

## Quick Start

### Requirements

- Python 3.12+
- NumPy, SciPy, Pandas
- matplotlib (for figures)

### Installation

```bash
# Clone the repository
git clone https://github.com/JoeLoera/liberation-day-replication.git
cd liberation-day-replication

# Install dependencies
pip install -r code_python/requirements.txt
```

### Important: Large Data Files

Due to GitHub file size limitations, **6GB+ of data files are not included**. See [DATA_README.md](DATA_README.md) for instructions on obtaining:
- Sectoral tariffs data (5.2 GB)
- BACI trade data (348 MB)
- ITPD database (966 MB)
- Dynamic Gravity data (125 MB)

### Running the Analysis

```bash
# Run all analyses (generates Tables 1-3, 8-10)
python3 run_all_python.py

# Run individual analyses
python3 code_python/analysis/main_baseline.py    # Tables 1-3, 9
python3 code_python/analysis/main_regional.py    # Table 8
python3 code_python/analysis/main_deficit.py     # Table 10
python3 code_python/analysis/main_io.py          # Tables 4, 7, 11 (in progress)
```

### Output

Results are saved to `python_output/`:
- LaTeX tables: `Table_1.tex`, `Table_2.tex`, etc.
- CSV files: `output_map.csv`, `output_map_retal.csv`
- Figures: `figure_1.png`

## Repository Structure

```
.
├── README.md                    # This file
├── DATA_README.md               # Instructions for large data files
├── code/                        # Original MATLAB code
├── code_python/                 # Python conversion
│   ├── analysis/                # Main analysis scripts
│   ├── utils/                   # Helper functions
│   └── requirements.txt         # Python dependencies
├── data/                        # Data files (see DATA_README.md)
├── output/                      # MATLAB reference outputs
├── python_output/               # Python generated outputs
│   └── REPLICATION_NOTES.md     # Detailed technical documentation
└── run_all_python.py            # Main execution script
```

## Key Conversion Fixes

The Python conversion required fixing several systematic errors in the MATLAB→NumPy axis mapping:

1. **Axis direction errors**: MATLAB `sum(X,2)` → NumPy `axis=1`
2. **Deficit calculation**: Corrected imports - exports direction
3. **Optimal tariff delta**: Fixed row/column sum axes
4. **IO model bounds**: Added validation for negative values
5. **Tariff dimensions**: Fixed 195→194 country mismatch

See [python_output/REPLICATION_NOTES.md](python_output/REPLICATION_NOTES.md) for complete technical details.

## Known Issues

### Input-Output Model (Tables 4, 7, 11)

The IO model with roundabout production linkages has convergence issues:
- Equilibrium solver runs for 15+ minutes without output
- Likely numerical precision or initial condition problems
- **Status**: Under investigation

## Comparison with MATLAB

All completed tables match the MATLAB reference outputs in `output/`. Minor discrepancies in Table 10 (Cases 1 & 3) are documented and show internal mathematical consistency in the Python implementation.

## Contributing

This is a course project. For questions or issues:
- Open an issue on GitHub
- Contact: [Your contact info]

## Original MATLAB Package

The original MATLAB replication package is included in the `code/` directory. See `readme.txt` for original documentation.

## License

[Specify license - typically same as original paper's replication package]

## Acknowledgments

- Original MATLAB code: Ignatenko, Macedoni, Lashkaripour, Simonovska
- Python conversion: [Your team members]
- Course: [Course name and instructor]

---

**Last Updated**: December 2025
