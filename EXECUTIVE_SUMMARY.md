# Executive Summary: Critical Axis Direction Bug in MATLAB-to-Python Conversion

## The Problem

The Python conversion of the trade model contains a **systematic axis direction error** that causes completely incorrect results. The axes for sum operations have been **swapped** in multiple critical locations throughout the codebase.

## Root Cause

The conversion process misunderstood how MATLAB dimensions map to NumPy axes:

- **MATLAB uses 1-based dimension indexing** where dimension 1 = columns, dimension 2 = rows
- **NumPy uses 0-based axis indexing** where axis 0 = rows, axis 1 = columns
- The converter appears to have incorrectly mapped MATLAB dim 1 → numpy axis 1 (WRONG)
- The correct mapping is: **MATLAB dim 1 → numpy axis 0, MATLAB dim 2 → numpy axis 1**

## The Correct Mapping (VERIFIED)

```
┌──────────────────┬───────────────────┬──────────────────────┐
│ MATLAB           │ NumPy             │ Result               │
├──────────────────┼───────────────────┼──────────────────────┤
│ sum(X, 1)        │ np.sum(X, axis=0) │ Column totals        │
│ sum(X, 2)        │ np.sum(X, axis=1) │ Row totals           │
└──────────────────┴───────────────────┴──────────────────────┘
```

### Trade Matrix Context

For trade matrix `X_ji` where **rows = exporters, columns = importers**:

- **IMPORTS** (total received by country i) = **column sums** = `sum(X_ji, 1)'` in MATLAB = `np.sum(X_ji, axis=0)` in NumPy
- **EXPORTS** (total sent by country i) = **row sums** = `sum(X_ji, 2)` in MATLAB = `np.sum(X_ji, axis=1)` in NumPy

## Critical Bugs Found

### Bug #1: GDP Calculation (Y_i)

**MATLAB line 37 in main_baseline.m:**
```matlab
Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';
```

**Current WRONG Python (line 188-189):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**Should be:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

**Impact:** Incorrect GDP values fed into all subsequent calculations.

---

### Bug #2: Equilibrium Condition (ERR1)

**MATLAB line 329 in main_baseline.m:**
```matlab
ERR1 = sum((1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;
```

**Current WRONG Python (lines 88-89):**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**Should be:**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

**Impact:** The solver tries to satisfy **wrong equilibrium conditions**, leading to convergence to incorrect solutions.

## Scope of Impact

### Affected Files (12-15 locations total):
1. `code_python/analysis/main_baseline.py` (2 bugs)
2. `code_python/analysis/main_deficit.py` (3 bugs)
3. `code_python/analysis/main_regional.py` (3 bugs)
4. `code_python/analysis/main_io.py` (2+ bugs)
5. `code_python/diagnostic_comparison.py` (1 bug)

### Consequences:
- ✗ GDP (Y_i) calculated incorrectly
- ✗ Expenditure (E_i) calculated incorrectly
- ✗ Equilibrium conditions (ERR1) are wrong
- ✗ Solver converges to wrong equilibrium
- ✗ All welfare calculations are wrong
- ✗ All trade flow predictions are wrong
- ✗ Results don't match MATLAB reference output

## Verification

Three test scripts confirm the bug:

1. **test_axis_mapping.py** - Verifies general axis mapping rules
2. **test_axis_detailed.py** - Proves Y_i calculation is wrong
3. **test_err1.py** - Proves ERR1 calculation is wrong

All tests show that:
- MATLAB `sum(X, 1)` = NumPy `sum(X, axis=0)` ✓
- MATLAB `sum(X, 2)` = NumPy `sum(X, axis=1)` ✓
- Current Python code has these swapped ✗

## Fix Required

**Simple fix:** Change `axis=0` to `axis=1` (and vice versa) in ~12-15 locations following specific patterns.

See `FIXES_REQUIRED.md` for complete list of exact locations and corrections.

## Next Steps

1. ✓ **Investigation complete** - Root cause identified and verified
2. ⧗ **Apply fixes** - Update all affected files per `FIXES_REQUIRED.md`
3. ⧗ **Test** - Run test scripts and compare with MATLAB output
4. ⧗ **Validate** - Ensure Python results match MATLAB reference

## Documentation Provided

1. **EXECUTIVE_SUMMARY.md** (this file) - High-level overview
2. **AXIS_MAPPING_FINDINGS.md** - Detailed technical analysis
3. **AXIS_VISUAL_GUIDE.txt** - Visual diagrams and examples
4. **FIXES_REQUIRED.md** - Complete list of every fix needed
5. **test_axis_mapping.py** - General verification test
6. **test_axis_detailed.py** - Y_i calculation test
7. **test_err1.py** - ERR1 calculation test

## Conclusion

This is a **critical bug** that affects the core calculations of the model. The fix is straightforward but must be applied systematically across all affected files. Once corrected, the Python implementation should produce results that match the MATLAB reference output.

**The axes were systematically swapped during conversion. They need to be unswapped.**
