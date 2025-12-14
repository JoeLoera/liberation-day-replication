# Critical Bugs Found and Fixed in Multi-Sector Model

## Summary

Through systematic comparison with MATLAB code, I identified and fixed **5 critical bugs** in the multi-sector baseline model that were preventing convergence.

---

## Bug #1: Incorrect Y_i_multi Calculation (CRITICAL)

**Location**: `sub_multisector_baseline.py`, lines 221-228

**Problem**: The calculation incorrectly included `beta_3D` parameter.

**MATLAB (Correct - Line 48)**:
```matlab
Y_i_multi = sum( repmat((1-nu)',N,1).*sum(X_ji,3) , 2) + nu.*sum(sum(X_ji,1),3)';
```

**Python (WRONG - Before Fix)**:
```python
Y_i_multi = np.sum(np.sum((1 - nu_3D) * beta_3D * X_ji, axis=2), axis=1) + \
            np.sum(np.sum(nu_3D * X_ji, axis=0), axis=1)
```

**Python (CORRECT - After Fix)**:
```python
Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
            nu * np.sum(np.sum(X_ji, axis=0), axis=1)
```

**Impact**: This bug caused Y_i values to be completely wrong, leading to equilibrium errors of ~10 million (vs expected < 1e-6).

---

## Bug #2: Incorrect Y_ik Calculation (CRITICAL)

**Location**: `sub_multisector_baseline.py`, lines 235-243

**Problem**: Similar to Bug #1, incorrectly included `beta_3D`.

**MATLAB (Correct - Lines 55-57)**:
```matlab
Y_ik_p = sum( repmat((1-nu)',[ N 1 K]).* X_ji , 2);
Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1);
Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3]);
```

**Python (WRONG - Before Fix)**:
```python
Y_ik_p = np.sum((1 - nu_3D) * beta_3D * X_ji, axis=1, keepdims=True)
Y_ik_f = np.transpose(np.sum(nu_3D * X_ji, axis=0, keepdims=True), (1, 0, 2))
```

**Python (CORRECT - After Fix)**:
```python
Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
```

---

## Bug #3: Incorrect Welfare Calculation

**Location**: `sub_multisector_baseline.py`, lines 97-99

**Problem**: Used `Ec_i` instead of `E_i` in delta_i calculation.

**MATLAB (Correct - Lines 162-163)**:
```matlab
delta_i = E_i./(E_i - kappa*(1-tau_i).*Y_i/(1+kappa));
W_i_h = delta_i .* (E_i_h ./ P_i_h) + (1-delta_i).*(w_i_h.*L_i_h ./ P_i_h);
```

**Python (WRONG - Before Fix)**:
```python
Ec_i = Y_i + T_i
delta_i = Ec_i / (Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
W_i_h = delta_i * (Ec_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)
```

**Python (CORRECT - After Fix)**:
```python
delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)
```

**Impact**: This bug caused welfare changes to be completely wrong (e.g., -7.13% instead of expected 0.60%).

---

## Bug #4: Incorrect ERR1 Normalization Equation

**Location**: `sub_multisector_baseline.py`, line 72

**Problem**: Used `sum()` instead of `mean()` for the replacement equation.

**MATLAB (Correct - Line 143)**:
```matlab
ERR1(N,1) = mean((P_i_h-1).*E_i);
```

**Python (WRONG - Before Fix)**:
```python
ERR1[N-1] = np.sum((E_i_h - 1) * E_i)
```

**Python (CORRECT - After Fix)**:
```python
ERR1[N-1] = np.mean((P_i_h - 1) * E_i)
```

**Impact**: This affected the scaling of the normalization equation, potentially causing convergence issues.

---

## Bug #5: Wrong tau_i Denominator

**Location**: `sub_multisector_baseline.py`, line 80

**Problem**: Used baseline `Y_i` instead of new `Y_i_new`.

**MATLAB (Correct - Line 132)**:
```matlab
tau_i = tariff_rev./Y_i_new;
```

**Python (WRONG - Before Fix)**:
```python
tau_i = tariff_rev / Y_i
```

**Python (CORRECT - After Fix)**:
```python
tau_i = tariff_rev / Y_i_new
```

**Impact**: This bug prevented the labor supply equation from calculating correctly during iterations.

---

## Diagnostic Results After Fixes

### Data Validation Test Results:
```
N = 194, K = 4
E_i_multi: sum = 1.26e+08
Y_i_multi: sum = 1.26e+08
T = E_i - Y_i: sum = 7.37e+04  (0.06% of total - ✓ Excellent!)
```

The sum(T) ≈ 0 confirms that E_i and Y_i are now correctly balanced.

---

## Remaining Issue: Slow Solver Convergence

**Status**: The solver is now getting the equations correct but converging very slowly (>2 minutes without completing).

**Possible Causes**:
1. Poor initial guess (all ones may be far from equilibrium)
2. Solver algorithm not optimal for this system (Python's hybr vs MATLAB's trust-region-dogleg)
3. System may have multiple equilibria
4. Need better scaling/normalization of equations

**Next Steps**:
1. Try better initial guesses (e.g., use baseline values as starting point)
2. Test different solver methods in scipy.optimize
3. Add equation scaling/normalization
4. Compare intermediate solver iterations with MATLAB

---

## Impact on Tables

These fixes are essential for:
- **Table 4**: Multi-sector comparison (rows 3 and 6)
- **Table 11**: Multi-sector and IO extensions (all rows)

Once the solver converges properly, these tables should match MATLAB exactly.
