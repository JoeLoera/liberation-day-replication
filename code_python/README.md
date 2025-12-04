# Python Conversion of Trade Model

This directory contains Python conversions of the MATLAB replication package for:

**"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"**
by Anna Ignatenko, Luca Macedoni, Ahmad Lashkaripour, Ina Simonovska (2025)

## Status of Conversion

### Completed:
- ✅ `utils/solver_utils.py` - Common solver functions (solve_nu, eq_fun)
- ✅ `analysis/main_baseline.py` - Baseline trade model
- ✅ `analysis/sub_multisector_baseline.py` - Multi-sector extension

### In Progress:
- 🔄 `analysis/print_tables_baseline.py` - LaTeX table generation
- 🔄 `analysis/main_io.py` - Input-output model
- 🔄 `analysis/main_regional.py` - Regional trade war analysis
- 🔄 `analysis/main_deficit.m` - Alternative deficit framework

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Verify data files are in place:
- `../../data/base_data/trade_cepii.csv`
- `../../data/base_data/gdp.csv`
- `../../data/base_data/tariffs.csv`
- `../../data/ITPDS/trade_ITPD.csv`

## Usage

### Run Baseline Analysis

```bash
cd code_python/analysis
python main_baseline.py
```

This will:
- Read trade and GDP data
- Solve for equilibrium under various tariff scenarios
- Generate output files in `../../output/`
- Print results to console

## Key Differences from MATLAB

### Indexing
- **MATLAB**: 1-indexed (id_US = 185)
- **Python**: 0-indexed (id_US = 184)

### Arrays
- **MATLAB**: `repmat()` → **Python**: `np.tile()`
- **MATLAB**: `eye(N)` → **Python**: `np.eye(N)`
- **MATLAB**: `.` → **Python**: `*` (element-wise multiply)

### Solvers
- **MATLAB**: `fsolve()` with optimoptions
- **Python**: `scipy.optimize.fsolve()` with xtol parameter

### File I/O
- **MATLAB**: `readtable()` → **Python**: `pd.read_csv()`
- **MATLAB**: `table2array()` → **Python**: `.values`

## Model Structure

### Baseline Model (`main_baseline.py`)

The baseline model solves for general equilibrium under:
- N = 194 countries
- Various tariff scenarios:
  1. USTR tariffs + income tax relief + no retaliation
  2. Partial pass-through
  3. Eaton-Kortum specification
  4. Optimal tariff without retaliation
  5. Liberation tariffs with optimal retaliation
  6. Liberation tariffs with reciprocal retaliation
  7. Optimal tariff with optimal retaliation
  8. Lump-sum rebate
  9. Higher trade elasticity

### Multi-Sector Model (`sub_multisector_baseline.py`)

Extension with:
- K = 4 sectors
- Sectoral reallocation of labor
- Sector-specific trade elasticities

## Validation

To validate Python outputs against MATLAB:

1. Run MATLAB version:
```matlab
cd replication_package
run run_all_matlab.m
```

2. Run Python version:
```bash
cd replication_package/code_python/analysis
python main_baseline.py
```

3. Compare outputs:
- `output/output_map.csv`
- Console output for US welfare changes

## Next Steps

To complete the conversion:

1. **Convert remaining analysis files**:
   - `main_io.m` → `main_io.py` (Input-output linkages)
   - `main_regional.m` → `main_regional.py` (Regional trade wars)
   - `main_deficit.m` → `main_deficit.py` (Alternative deficit framework)

2. **Convert table printing**:
   - `print_tables_baseline.m` → `print_tables_baseline.py`
   - `print_tables_io.m` → `print_tables_io.py`

3. **Create master script**:
   - `run_all_python.py` to replicate `run_all_matlab.m`

4. **Validation**:
   - Compare all output tables
   - Verify numerical precision (should match to ~4 decimal places)

## Contact

For questions about the Python conversion:
- Check this README
- Compare with MATLAB originals in `../code/analysis/`
- Refer to original paper for model details

For questions about the replication package:
- Email: ahmadlp@gmail.com
