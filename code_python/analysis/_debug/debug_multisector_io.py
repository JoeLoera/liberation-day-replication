"""
Debug multi-sector IO model discrepancy
Target: -4.1% d_trade
Current: -6.1% d_trade

Compare key intermediate values with MATLAB
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
output_dir = os.path.join(base_path, "python_output")

print("=" * 80)
print("Debugging Multi-Sector IO Model")
print("=" * 80)

# Load sectoral trade data
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K), order='F')

print(f"\n1. TRADE DATA LOADING")
print(f"   Total trade volume: {np.sum(X_ji):.2e}")
print(f"   US (184) domestic trade by sector: {X_ji[184, 184, :]}")

# Check MATLAB approach: set tariffs BEFORE filtering
print(f"\n2. CHECKING MATLAB vs PYTHON TARIFF SETUP")

# Load tariffs (full 194-country array)
tariff_data_full = pd.read_csv(os.path.join(base_path, 'data', 'base_data', 'tariffs.csv'))
new_ustariff_full = tariff_data_full.values
print(f"   Full tariff array shape: {new_ustariff_full.shape}")
print(f"   Tariff on US (row 184): {new_ustariff_full[184, 0]:.4f}")

# MATLAB approach: Set tariffs on full array, then filter
id_US_full = 184  # 0-indexed
t_ji_matlab = np.zeros((N, N, K))
t_ji_matlab[:, id_US_full, :K-1] = np.tile(new_ustariff_full, (1, K-1))
t_ji_matlab[:, id_US_full, :K-1] = np.maximum(0.1, t_ji_matlab[:, id_US_full, :K-1])
t_ji_matlab[id_US_full, id_US_full, :K-1] = 0

# Filter countries
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N), ID)
N_filtered = len(idx)
print(f"   Filtered from {N} to {N_filtered} countries")
print(f"   Removed country indices: {ID}")

# Filter tariffs (MATLAB way)
t_ji_matlab_filtered = np.zeros((N_filtered, N_filtered, K))
for k in range(K):
    t_ji_matlab_filtered[:, :, k] = t_ji_matlab[np.ix_(idx, idx, [k])].reshape(N_filtered, N_filtered)

id_US_new = np.where(idx == id_US_full)[0][0]
print(f"   US index after filtering: {id_US_new}")

# Python current approach: Filter first, then set tariffs
new_ustariff_filtered = new_ustariff_full[idx, :]
t_ji_python = np.zeros((N_filtered, N_filtered, K))
t_ji_python[:, id_US_new, :K-1] = np.tile(new_ustariff_filtered, (1, K-1))
t_ji_python[:, id_US_new, :K-1] = np.maximum(0.1, t_ji_python[:, id_US_new, :K-1])
t_ji_python[id_US_new, id_US_new, :K-1] = 0

# Compare
print(f"\n3. TARIFF COMPARISON (MATLAB vs PYTHON approach)")
diff = np.abs(t_ji_matlab_filtered - t_ji_python)
print(f"   Max difference: {np.max(diff):.6f}")
print(f"   Mean difference: {np.mean(diff):.6f}")
if np.allclose(t_ji_matlab_filtered, t_ji_python):
    print("   ✓ Tariff setups are IDENTICAL")
else:
    print("   ✗ Tariff setups DIFFER!")
    # Find where they differ
    diff_idx = np.where(diff > 1e-10)
    if len(diff_idx[0]) > 0:
        print(f"   First difference at: {diff_idx[0][0]}, {diff_idx[1][0]}, {diff_idx[2][0]}")
        print(f"   MATLAB value: {t_ji_matlab_filtered[diff_idx[0][0], diff_idx[1][0], diff_idx[2][0]]}")
        print(f"   Python value: {t_ji_python[diff_idx[0][0], diff_idx[1][0], diff_idx[2][0]]}")

# Check phi values
print(f"\n4. PHI VALUES CHECK")
phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
phi_full = phi_data['phi'].values
print(f"   Full phi array shape: {phi_full.shape}")
print(f"   US phi (index 184): {phi_full[184]:.4f}")

# Load Y_i for phi_avg calculation
Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values
print(f"   Y_i_baseline shape: {Y_i_baseline.shape}")

# Note: MATLAB uses Phi{1} for phi_avg, which might be different from Phi{2}
# Let's check what phi_avg we're computing
phi_avg = np.sum(phi_full * Y_i_baseline) / np.sum(Y_i_baseline)
print(f"   Computed phi_avg: {phi_avg:.6f}")

# Check eps values
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
print(f"\n5. TRADE ELASTICITIES")
print(f"   eps (by sector): {eps}")

# Check nu values
print(f"\n6. NU VALUES CHECK")
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu_full = nu_data['nu'].values
print(f"   Full nu array shape: {nu_full.shape}")
print(f"   US nu (index 184): {nu_full[184]:.4f}")
print(f"   Non-US nu (index 0): {nu_full[0]:.4f}")

# Load the multi-sector IO results
print(f"\n7. CURRENT RESULTS")
try:
    results = np.load(os.path.join(output_dir, 'multisector_io_results.npz'))
    print(f"   d_trade_IO_multi[0] (no retaliation): {results['d_trade_IO_multi'][0]:.2f}%")
    print(f"   d_trade_IO_multi[1] (retaliation): {results['d_trade_IO_multi'][1]:.2f}%")
    print(f"   USA welfare (no retaliation): {results['results_multi'][results['id_US'], 0, 0]:.2f}%")
    print(f"   USA welfare (retaliation): {results['results_multi'][results['id_US'], 0, 1]:.2f}%")
except Exception as e:
    print(f"   Could not load results: {e}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS:")
print("=" * 80)
print("""
1. In MATLAB, tariffs are set BEFORE filtering countries, then the full
   t_ji array is filtered. This should be equivalent to Python's approach.

2. MATLAB uses Phi{1} for phi_avg calculation, but Phi{2} for actual phi.
   These could be different! Need to check the baseline model.

3. The trade elasticity eps depends on phi_avg. If phi_avg is wrong,
   eps will be wrong, affecting the equilibrium.

4. MATLAB uses 'levenberg-marquardt' with only 15 iterations for the first
   scenario. Python uses default fsolve which may converge differently.
""")
