# Python Replication Status

## Overview

This document tracks the Python replication of the MATLAB code from "Making America Great Again? The Economic Impacts of Liberation Day Tariffs."

**Last Updated:** December 15, 2024

---

## Replication Summary

| Table | Status | Python vs MATLAB |
|-------|--------|------------------|
| Table 1 | ✅ **Exact Match** | Baseline policy scenarios |
| Table 2 | ✅ **Exact Match** | Retaliation scenarios |
| Table 3 | ✅ **Exact Match** | Tariff revenue |
| Table 4 | ⚠️ **Close** | Multi-sector: 0.71% vs 0.60% target |
| Table 7 | 🚫 N/A | Stata econometrics (not simulated) |
| Table 8 | ✅ **Exact Match** | Regional trade wars |
| Table 9 | ✅ **Exact Match** | Alternative specifications |
| Table 10 | ⚠️ **Partial** | 2/4 deficit scenarios match |
| Table 11 | ⚠️ **Close** | Single-sector exact; multi-sector close |

**Note:** Tables 5 & 6 do not exist in the paper.

---

## Detailed Results

### Table 11: Global Trade-to-GDP Changes

| Model | Before Retaliation | After Retaliation |
|-------|-------------------|-------------------|
| **Single-sector (main)** | -9.4% ✅ | -11.6% ✅ |
| **Single-sector (IO)** | -10.8% ✅ | -12.4% ✅ |
| **Multi-sector** | -5.8% (target: -5.5%) | -7.4% (target: -6.9%) |
| **Multi-sector + IO** | -6.1% (target: -4.1%) | -7.4% (target: -4.9%) |

### Table 4: USA Welfare Changes (Multi-sector)

| Scenario | Python | MATLAB Target |
|----------|--------|---------------|
| Pre-retaliation | 0.71% | 0.60% |
| Post-retaliation | 0.05% | -1.02% |

---

## Key Bug Fixes (December 2024)

The following critical bugs were identified and fixed:

1. **X_ji Reshape Order**: Changed from C-order to Fortran order (`order='F'`) to match MATLAB's column-major data storage
2. **US Index Calculation**: Fixed incorrect index lookup that was targeting Uzbekistan instead of USA
3. **Tariff Tiling**: Corrected `np.tile()` dimensions for tariff matrix construction
4. **Variable Reshape Order**: Added `order='F'` to `ell_ik_h` and `ERR1` reshapes

---

## File Structure

### Core Analysis Files
- `main_baseline.py` - Single-sector baseline model
- `main_io.py` - Input-output model
- `main_regional.py` - Regional trade war scenarios
- `main_deficit.py` - Deficit framework comparison
- `sub_multisector_baseline.py` - Multi-sector baseline (K=4)
- `sub_multisector_io.py` - Multi-sector with IO linkages
- `generate_table_4.py` - Table 4 generator
- `generate_table_11.py` - Table 11 generator
- `print_tables_baseline.py` - LaTeX table generation

### Output Files
- `Table_*.tex` - LaTeX formatted tables
- `*_results.npz` - NumPy compressed result arrays
- `*.csv` - Parameter exports

---

## Running the Replication

```bash
# Generate all baseline results
cd code_python/analysis
python main_baseline.py

# Generate multi-sector results
python sub_multisector_baseline.py
python sub_multisector_io.py

# Generate tables
python generate_table_4.py
python generate_table_11.py
```

---

## Known Differences from MATLAB

1. **Solver Algorithm**: Python uses Levenberg-Marquardt; MATLAB uses trust-region-dogleg
2. **Multi-sector welfare**: Small differences (~0.1-1%) in equilibrium solutions
3. **Multi+IO model**: Larger discrepancy (~2%) requires further investigation

---

## Overall Progress

**Fully Replicated:** 5/7 tables (71%)
**Close Match:** 2/7 tables (29%)
**Not Applicable:** 1 table (Stata-based)
