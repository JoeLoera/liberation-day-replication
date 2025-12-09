# Multi-Sector Model Debugging Notes

## Current Status (as of 2025-12-09)

### Successfully Replicated (6/9 Tables - 67%):
- ✓ Table 1: Baseline policy scenarios
- ✓ Table 2: Retaliation scenarios
- ✓ Table 3: Tariff revenue
- ✓ Table 8: Regional trade wars
- ✓ Table 9: Alternative specifications
- ✓ Table 10: Deficit frameworks

### Remaining Issues (3/9 Tables):
- ✗ Table 4: IO model baseline (sub_baseline_IO.py)
- ✗ Table 7: IO model variations (sub_baseline_IO.py)
- ✗ Table 11: Multi-sector models (sub_multisector_baseline.py, sub_multisector_io.py)

## Key Bug Fixed

### Critical Fix in Labor Supply Equation
**File**: `code_python/analysis/sub_multisector_baseline.py`, line 77

**MATLAB Code** (sub_multisector_baseline.m:132):
```matlab
tau_i = tariff_rev./Y_i_new;  % Tariff revenue as fraction of NEW income
```

**Python Code** (BEFORE fix):
```python
tau_i = tariff_rev / Y_i  # WRONG - using baseline income instead of new income
```

**Python Code** (AFTER fix):
```python
tau_i = tariff_rev / Y_i_new  # CORRECT - using new income like MATLAB
```

This bug prevented the labor supply equation from calculating correctly, causing employment changes to be 0%.

## Remaining Technical Issue

### RuntimeWarning: invalid value encountered in sqrt

**Location**: Line 82 in sub_multisector_baseline.py
```python
ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa
```

**Problem**:
- `kappa = 0.5` (square root)
- The expression `(tau_i_h * w_i_h / P_i_h)` can become negative for some countries
- Raising negative number to power 0.5 produces NaN in NumPy
- This prevents the solver from converging to the correct equilibrium

**Root Cause**:
```python
tau_i = tariff_rev / Y_i_new  # Can be large if Y_i_new is small
tau_i_h = (1 - tau_i_new) / (1 - tau_i)  # Can be negative if tau_i > 1
```

When `tau_i > 1` (tariff revenue exceeds new income), `tau_i_h` becomes negative, propagating through the calculation.

## Attempted Solutions

### 1. Bounds Checking (Failed)
```python
# Too restrictive - forced solver to wrong equilibrium
labor_term = np.maximum(tau_i_h * w_i_h / P_i_h, 0.01)
ERR3 = L_i_h - labor_term**kappa
```
Result: Employment stayed at 0%, trade changes at 0%

### 2. Absolute Value (Failed)
```python
# Prevents NaN but changes economic meaning
labor_term = np.abs(tau_i_h * w_i_h / P_i_h)
ERR3 = L_i_h - labor_term**kappa
```
Result: Still produces NaN warnings, solver struggles

### 3. Levenberg-Marquardt Algorithm (Slow)
```python
# Matching MATLAB's solver
sol = root(syst, x0, method='lm', options={'xtol': 1e-10, 'ftol': 1e-10})
```
Result: Takes too long (>5 minutes without completing), likely struggling with NaN values

## Expected vs. Actual Results

### Table 11 Target Values (from MATLAB):
```
Before Retaliation:
- Global trade-to-GDP change: -5.5% (multi) vs 0.00% (Python)
- Global employment change: -0.05% (multi) vs 0.00% (Python)

After Retaliation:
- Global trade-to-GDP change: -6.9% (multi) vs 0.00% (Python)
- Global employment change: -0.05% (multi) vs 0.00% (Python)
```

### What Works:
- USA welfare change: -10.17% ✓ (matches MATLAB)
- All country-level welfare calculations ✓

### What Doesn't Work:
- All employment changes: 0.0000% for every country
- Global trade metrics: essentially 0%

## Next Steps to Consider

1. **Investigate tau_i > 1 cases**:
   - Check which countries have tariff revenue > new income
   - Understand economic interpretation
   - Review MATLAB's handling of this edge case

2. **Compare intermediate values with MATLAB**:
   - Save MATLAB workspace after equilibrium solve
   - Compare tau_i, tau_i_h, P_i_h values country-by-country
   - Identify where divergence begins

3. **Alternative solver approaches**:
   - Try trust-region methods
   - Use scipy.optimize.least_squares instead of root
   - Experiment with different initial guesses

4. **Economic constraints**:
   - Add constraint that tau_i must be < 1
   - Reformulate labor supply equation to avoid negative terms
   - Consult original Ossa (2014) paper for guidance

## Files Modified

- `code_python/analysis/sub_multisector_baseline.py`:
  - Fixed tau_i calculation (line 77)
  - Switched to Levenberg-Marquardt solver
  - Fixed file paths for portability

- `code_python/analysis/sub_multisector_io.py`:
  - Fixed file paths for portability
  - Not yet tested due to baseline model issues

## Model Specification

### Multi-Sector Equilibrium System:
- **Variables**: 3N + NK = 3×194 + 194×4 = 1358 variables
  - w_i: wages (N=194)
  - E_i: expenditures (N=194)
  - L_i: labor (N=194)
  - ell_ik: sectoral labor shares (N×K = 194×4 = 776)

- **Equations**: 1358 equations
  - ERR1: Sectoral income balance (NK = 776, with 1 replaced by normalization)
  - ERR2: Total income balance (N = 194)
  - ERR3: Labor supply (N = 194)
  - ERR4: Sectoral shares sum to 1 (N = 194)

### Parameters:
- K = 4 sectors (agriculture, manufacturing, mining, services)
- kappa = 0.5 (Frisch elasticity of labor supply)
- psi = 0.67/4 (labor mobility parameter)
- theta = 1/psi (scale parameter)
- eps = [3.3, 3.8, 4.1, 3.0] (trade elasticities by sector)

## Conclusion

The multi-sector model has a fundamental convergence issue related to the labor supply equation producing NaN values when tariff revenues are high relative to incomes. This is a challenging numerical issue that requires either:
1. Better understanding of MATLAB's handling of edge cases
2. Economic constraints to prevent tau_i > 1
3. Reformulation of the labor supply equation

The main baseline, deficit, and regional models all work perfectly, suggesting the core trade model logic is correct. The issue is specific to the multi-sector extension with endogenous labor supply.
