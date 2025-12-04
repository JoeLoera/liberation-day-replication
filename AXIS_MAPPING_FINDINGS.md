# MATLAB-to-Python Axis Direction Bug - Complete Analysis

## Executive Summary

**CRITICAL FINDING:** The Python conversion has **systematically reversed** the axis parameters in multiple locations throughout the codebase. This is causing completely incorrect results.

## Axis Mapping Rules (VERIFIED)

### MATLAB Behavior:
- `sum(X, 1)` - Sums DOWN columns (collapses rows) → Gives column totals (1×N row vector)
- `sum(X, 2)` - Sums ACROSS rows (collapses columns) → Gives row totals (N×1 column vector)

### NumPy Behavior:
- `np.sum(X, axis=0)` - Sums DOWN axis 0 (collapses rows) → Gives column totals
- `np.sum(X, axis=1)` - Sums ACROSS axis 1 (collapses columns) → Gives row totals

### Correct Mapping:
```
MATLAB sum(X, 1) ≡ numpy sum(X, axis=0)  [COLUMN totals / imports]
MATLAB sum(X, 2) ≡ numpy sum(X, axis=1)  [ROW totals / exports]
```

## Trade Matrix Context

For trade matrix X_ji where **rows = exporters, columns = importers**:

- **Total IMPORTS** by country i = sum over all exporters j = **column sums**
  - MATLAB: `sum(X_ji, 1)'` (transpose to column vector)
  - Python: `np.sum(X_ji, axis=0)`

- **Total EXPORTS** by country i = sum over all importers j = **row sums**
  - MATLAB: `sum(X_ji, 2)` (already column vector)
  - Python: `np.sum(X_ji, axis=1)`

## Bugs Found

### Bug 1: Y_i Calculation (Line 188-189 in main_baseline.py)

**MATLAB (line 37 in main_baseline.m):**
```matlab
Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';
```

**Current INCORRECT Python (lines 188-189):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**CORRECT Python:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

**Issue:** First term uses `axis=0` (column sums) but should use `axis=1` (row sums) to match MATLAB's `sum(..., 2)`.

### Bug 2: ERR1 Calculation (Lines 88-89 in main_baseline.py)

**MATLAB (line 329 in main_baseline.m):**
```matlab
ERR1 = sum((1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;
```

**Current INCORRECT Python (lines 88-89):**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**CORRECT Python:**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

**Issue:** The axes are **completely swapped**:
- First term uses `axis=0` but should use `axis=1` (to match MATLAB's `sum(..., 2)`)
- Second term uses `axis=1` but should use `axis=0` (to match MATLAB's `sum(..., 1)`)

### Bug 3: T Calculation (Line 177-178 in main_baseline.py)

**MATLAB (line 29 in main_baseline.m):**
```matlab
T = (1-nu).*(sum(X_ji,1)' - sum(repmat((1-nu)',N,1).*X_ji,2));
```

**Current Python (lines 177-178):**
```python
T = (1 - nu) * (np.sum(X_ji, axis=0) -
                np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1))
```

**Status:** This one is actually **CORRECT**!
- First term: `axis=0` matches MATLAB's `sum(X_ji,1)'` (column sums)
- Second term: `axis=1` matches MATLAB's `sum(...,2)` (row sums)

## Affected Files

The same bugs appear in multiple files:

1. **main_baseline.py** - Lines 88-89 (ERR1), Line 188-189 (Y_i)
2. **main_deficit.py** - Lines 66-67 (ERR1), Line 148-149 (Y_i)
3. **main_regional.py** - Lines 66-67 (ERR1), Line 156-157 (Y_i)
4. **main_io.py** - Lines 82-83 (ERR1), possibly others
5. **diagnostic_comparison.py** - Line 90 (Y_i)

## Verification Tests

Three test scripts were created to verify the findings:

1. `test_axis_mapping.py` - General axis mapping verification
2. `test_axis_detailed.py` - Detailed test for Y_i calculation (line 37)
3. `test_err1.py` - Detailed test for ERR1 calculation (line 329)

All tests confirm the axis direction errors.

## Impact

These axis errors cause:
- Incorrect calculation of GDP (Y_i)
- Incorrect equilibrium conditions (ERR1)
- Solver convergence to wrong solutions
- Results that don't match MATLAB reference output

## Recommended Fix

Search for all instances of the following patterns and swap the axes:

1. Pattern: `np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0)`
   - Should be: `axis=1` when matching MATLAB's `sum(repmat((1-nu)',N,1).*X_ji,2)`

2. Pattern: `np.sum((1 - nu_2D) * X_ji_new, axis=0) + np.sum(nu_2D * X_ji_new, axis=1)`
   - Should be: `np.sum((1 - nu_2D) * X_ji_new, axis=1) + np.sum(nu_2D * X_ji_new, axis=0)`

3. Similar patterns with beta coefficient in main_io.py

## Files Requiring Fixes

Based on grep search:

### ERR1 fixes needed:
- `code_python/analysis/main_baseline.py` (line 88-89)
- `code_python/analysis/main_deficit.py` (line 66-67)
- `code_python/analysis/main_regional.py` (line 66-67)
- `code_python/analysis/main_io.py` (line 82-83)

### Y_i calculation fixes needed:
- `code_python/analysis/main_baseline.py` (line 188-189)
- `code_python/analysis/main_deficit.py` (line 148-149)
- `code_python/analysis/main_regional.py` (line 156-157)
- `code_python/analysis/main_io.py` (possibly multiple locations)
- `code_python/diagnostic_comparison.py` (line 90)

### T calculation fixes needed:
- `code_python/analysis/main_io.py` (line 329)
- `code_python/analysis/main_regional.py` (line 146)
- `code_python/analysis/main_deficit.py` (line 138)

## Root Cause

The conversion process appears to have made a systematic error in understanding MATLAB's dimension indexing:
- MATLAB uses 1-based dimension indexing (1 = columns, 2 = rows)
- NumPy uses 0-based axis indexing (0 = rows, 1 = columns)
- The converter appears to have incorrectly mapped MATLAB dim 1 → numpy axis 1, and MATLAB dim 2 → numpy axis 0
- The correct mapping is: MATLAB dim 1 → numpy axis 0, and MATLAB dim 2 → numpy axis 1
