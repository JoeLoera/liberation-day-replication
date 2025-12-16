"""
Verify the bug in Y_i_multi calculation
MATLAB: Y_i_multi = sum( repmat((1-nu)',N,1).*sum(X_ji,3) , 2) + nu.*sum(sum(X_ji,1),3)'
Python (WRONG): Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + ...
Python (CORRECT): Y_i_multi = np.sum((1 - nu).reshape(1, -1) * np.sum(X_ji, axis=2), axis=1) + ...

The bug: reshape(-1, 1) creates (N, 1) which broadcasts as nu_j
         reshape(1, -1) creates (1, N) which broadcasts as nu_i
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

# Load data
print("Loading data...")
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K))

# Remove problematic countries
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N), ID)
N = len(idx)

X_new = np.zeros((N, N, K))
for k in range(K):
    X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
X_ji = X_new

# Load nu
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

print(f"N = {N}")
print(f"nu range: [{nu.min():.4f}, {nu.max():.4f}]")

# ============================================================
# WRONG calculation (current code): reshape(-1, 1) = (N, 1)
# ============================================================
print("\n" + "=" * 60)
print("WRONG calculation: reshape(-1, 1)")
print("=" * 60)

# (1 - nu).reshape(-1, 1) has shape (N, 1)
# When multiplied with (N, N), element (j, i) = (1 - nu[j])  <-- WRONG!
Y_i_multi_WRONG = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
                  nu * np.sum(np.sum(X_ji, axis=0), axis=1)

# Compute Y_ik (this is correct in both versions)
Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))

sum_Y_ik = np.sum(Y_ik, axis=2).flatten()
ell_ik_WRONG = Y_ik / np.tile(Y_i_multi_WRONG.reshape(-1, 1, 1), (1, 1, K))
sum_ell_ik_WRONG = np.sum(ell_ik_WRONG, axis=2).flatten()

print(f"Y_i_multi_WRONG range: [{Y_i_multi_WRONG.min():.2e}, {Y_i_multi_WRONG.max():.2e}]")
print(f"sum(Y_ik) range: [{sum_Y_ik.min():.2e}, {sum_Y_ik.max():.2e}]")
print(f"Ratio sum(Y_ik)/Y_i_multi: [{(sum_Y_ik/Y_i_multi_WRONG).min():.4f}, {(sum_Y_ik/Y_i_multi_WRONG).max():.4f}]")
print(f"sum(ell_ik) range: [{sum_ell_ik_WRONG.min():.4f}, {sum_ell_ik_WRONG.max():.4f}]")
print(f"ERR4 at x=1 (100*(sum-1)): [{100*(sum_ell_ik_WRONG.min()-1):.2f}, {100*(sum_ell_ik_WRONG.max()-1):.2f}]")

# ============================================================
# CORRECT calculation: reshape(1, -1) = (1, N)
# ============================================================
print("\n" + "=" * 60)
print("CORRECT calculation: reshape(1, -1)")
print("=" * 60)

# (1 - nu).reshape(1, -1) has shape (1, N)
# When multiplied with (N, N), element (j, i) = (1 - nu[i])  <-- CORRECT!
Y_i_multi_CORRECT = np.sum((1 - nu).reshape(1, -1) * np.sum(X_ji, axis=2), axis=1) + \
                    nu * np.sum(np.sum(X_ji, axis=0), axis=1)

ell_ik_CORRECT = Y_ik / np.tile(Y_i_multi_CORRECT.reshape(-1, 1, 1), (1, 1, K))
sum_ell_ik_CORRECT = np.sum(ell_ik_CORRECT, axis=2).flatten()

print(f"Y_i_multi_CORRECT range: [{Y_i_multi_CORRECT.min():.2e}, {Y_i_multi_CORRECT.max():.2e}]")
print(f"sum(Y_ik) range: [{sum_Y_ik.min():.2e}, {sum_Y_ik.max():.2e}]")
print(f"Ratio sum(Y_ik)/Y_i_multi: [{(sum_Y_ik/Y_i_multi_CORRECT).min():.6f}, {(sum_Y_ik/Y_i_multi_CORRECT).max():.6f}]")
print(f"sum(ell_ik) range: [{sum_ell_ik_CORRECT.min():.6f}, {sum_ell_ik_CORRECT.max():.6f}]")
print(f"ERR4 at x=1 (100*(sum-1)): [{100*(sum_ell_ik_CORRECT.min()-1):.6f}, {100*(sum_ell_ik_CORRECT.max()-1):.6f}]")

# ============================================================
# Verify the fix
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

if np.allclose(sum_ell_ik_CORRECT, 1.0, atol=1e-6):
    print("SUCCESS: With CORRECT formula, sum(ell_ik) = 1.0 for all countries!")
    print("This should make ERR4 = 0 at the initial guess (x = all ones)")
else:
    print(f"ISSUE: sum(ell_ik) still not exactly 1.0")
    print(f"Max deviation: {np.max(np.abs(sum_ell_ik_CORRECT - 1.0)):.2e}")

# Show the difference between WRONG and CORRECT
diff = np.abs(Y_i_multi_CORRECT - Y_i_multi_WRONG)
print(f"\nDifference |Y_i_correct - Y_i_wrong|:")
print(f"  min: {diff.min():.2e}")
print(f"  max: {diff.max():.2e}")
print(f"  mean: {diff.mean():.2e}")
