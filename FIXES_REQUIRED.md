# Complete List of Axis Direction Fixes Required

## Summary

The MATLAB-to-Python conversion has systematically swapped axes in calculations involving `repmat((1-nu)',N,1)` and `nu_2D` patterns. This document lists every location requiring a fix.

## Mapping Rule

```
MATLAB sum(X, 1) → NumPy np.sum(X, axis=0)  [column totals]
MATLAB sum(X, 2) → NumPy np.sum(X, axis=1)  [row totals]
```

---

## File 1: code_python/analysis/main_baseline.py

### Fix 1: Lines 88-89 (ERR1 calculation)

**Current (WRONG):**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**Corrected:**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

**Reason:** MATLAB line 329 has `sum(...,2) + sum(...,1)'` which maps to `axis=1 + axis=0`

---

### Fix 2: Lines 188-189 (Y_i calculation)

**Current (WRONG):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**Corrected:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

**Reason:** MATLAB line 37 has `sum(repmat((1-nu)',N,1).*X_ji,2)` which uses dimension 2 → `axis=1`

---

## File 2: code_python/analysis/main_deficit.py

### Fix 1: Lines 66-67 (ERR1 calculation)

**Current (WRONG):**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**Corrected:**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

---

### Fix 2: Line 138 (T calculation)

**Current (WRONG):**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0)
```

**Corrected:**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1)
```

**Reason:** This appears in a T calculation matching MATLAB's `sum(repmat((1-nu)',N,1).*X_ji,2)`

---

### Fix 3: Lines 148-149 (Y_i calculation)

**Current (WRONG):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**Corrected:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

---

## File 3: code_python/analysis/main_regional.py

### Fix 1: Lines 66-67 (ERR1 calculation)

**Current (WRONG):**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**Corrected:**
```python
ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

---

### Fix 2: Line 146 (T calculation)

**Current (WRONG):**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0)
```

**Corrected:**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1)
```

---

### Fix 3: Lines 156-157 (Y_i calculation)

**Current (WRONG):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**Corrected:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

---

## File 4: code_python/analysis/main_io.py

### Fix 1: Lines 82-83 (ERR1 calculation with beta)

**Current (WRONG):**
```python
ERR1 = np.sum(beta * (1 - nu_2D) * X_ji_new, axis=0) + \
       np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i
```

**Corrected:**
```python
ERR1 = np.sum(beta * (1 - nu_2D) * X_ji_new, axis=1) + \
       np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i
```

**Note:** This file uses a `beta` coefficient but the same axis swap applies.

---

### Fix 2: Line 329 (T calculation)

Check the context around line 329 for the T calculation pattern. Based on grep results, this likely needs:

**Current (WRONG):**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0)
```

**Corrected:**
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1)
```

---

### Fix 3: Additional Y_i calculations

Search the file for other instances of the Y_i calculation pattern and apply the same fix (change `axis=0` to `axis=1` in the first term).

---

## File 5: code_python/diagnostic_comparison.py

### Fix 1: Line 72 (appears correct based on grep)

This line already uses `axis=1` which is correct:
```python
np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1)
```
**No change needed.**

---

### Fix 2: Line 90 (Y_i calculation)

**Current (WRONG):**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) + \
      nu * np.sum(X_ji, axis=0)
```

**Corrected:**
```python
Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) + \
      nu * np.sum(X_ji, axis=0)
```

---

## Verification Strategy

After making fixes:

1. **Run test scripts** to verify axis behavior:
   - `test_axis_mapping.py`
   - `test_axis_detailed.py`
   - `test_err1.py`

2. **Compare Python output** to MATLAB reference output:
   - Check if E_i values match
   - Check if Y_i values match
   - Check if welfare calculations match

3. **Run full model** and compare final results with MATLAB tables

---

## Search Patterns to Find All Instances

Use these grep commands to verify all instances have been fixed:

```bash
# Find all ERR1 patterns
grep -n "ERR1 = np.sum((1 - nu_2D)" code_python/analysis/*.py

# Find all Y_i calculation patterns
grep -n "Y_i = np.sum(np.tile((1 - nu)" code_python/analysis/*.py

# Find all T calculation patterns
grep -n "np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji" code_python/**/*.py
```

---

## Summary Statistics

- **Total files affected:** 5
- **Total fixes required:** ~12-15 locations
- **Pattern types:**
  - ERR1 calculations: 4 instances
  - Y_i calculations: 5 instances
  - T calculations: 3 instances

---

## Testing After Fixes

After applying all fixes, run:

```bash
python code_python/test_conversion.py
python code_python/diagnostic_comparison.py
python code_python/analysis/main_baseline.py
```

Compare the output with MATLAB reference files in `output/` directory.
