"""
Test least_squares solver for multi-sector model
This uses scipy.optimize.least_squares with trust-region algorithm
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import sys
import os
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sub_multisector_baseline import balanced_trade_multisector

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

print("="*80)
print("Testing scipy.optimize.least_squares (Trust-Region Reflective)")
print("="*80)

# Load sectoral trade data
print("\nLoading sectoral trade data...")
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K))

# Remove countries with no trade
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N), ID)
N = len(idx)

X_new = np.zeros((N, N, K))
for k in range(K):
    X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
X_ji = X_new

# Load parameters
phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
phi = phi_data['phi'].values[idx]
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

# Calculate initial values
E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
            nu * np.sum(np.sum(X_ji, axis=0), axis=1)
T = E_i_multi - Y_i_multi

# Calculate parameters
lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
         np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

# Load tariffs
tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
tariff_data_full = pd.read_csv(tariff_path)
new_ustariff_full = tariff_data_full.values
new_ustariff = new_ustariff_full[idx, :]

id_US = 185 - 1
id_US_new = np.where(idx == id_US + 1)[0][0]

t_ji = np.zeros((N, N, K))
t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
t_ji[id_US_new, id_US_new, :K-1] = 0

# Trade elasticities
Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

# Set up data and params
data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
param = [eps_3D, kappa, psi, phi]

# Initial guess: all ones (matching MATLAB)
x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

def syst(x):
    ceq, _, _ = balanced_trade_multisector(x, data, param)
    return ceq

print(f"\nSystem size: {len(x0)} variables, {len(x0)} equations")
print(f"Initial error: {np.max(np.abs(syst(x0))):.2e}")

# Test least_squares with trust-region method
print("\n" + "-"*80)
print("Solving with least_squares (trf method)...")
print("-"*80)

start_time = time.time()

# Use trust-region reflective method (best for large-scale problems)
result = least_squares(
    syst,
    x0,
    method='trf',  # Trust-Region Reflective (handles bounds, large-scale)
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    max_nfev=10000,
    verbose=2  # Show iteration progress
)

elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Function evaluations: {result.nfev}")
print(f"Jacobian evaluations: {result.njev}")
print(f"Runtime: {elapsed_time:.1f} seconds")

# Check final error
ceq_final = syst(result.x)
print(f"\nFinal equilibrium error:")
print(f"  Max: {np.max(np.abs(ceq_final)):.2e}")
print(f"  Mean: {np.mean(np.abs(ceq_final)):.2e}")
print(f"  Median: {np.median(np.abs(ceq_final)):.2e}")

# Break down by equation type
ERR1 = ceq_final[:N*K]
ERR2 = ceq_final[N*K:N*K+N]
ERR3 = ceq_final[N*K+N:N*K+2*N]
ERR4 = ceq_final[N*K+2*N:]

print(f"\nError breakdown:")
print(f"  ERR1 (sectoral income): max={np.max(np.abs(ERR1)):.2e}")
print(f"  ERR2 (total income): max={np.max(np.abs(ERR2)):.2e}")
print(f"  ERR3 (labor supply): max={np.max(np.abs(ERR3)):.2e}")
print(f"  ERR4 (sectoral shares): max={np.max(np.abs(ERR4)):.2e}")

if np.max(np.abs(ceq_final)) < 1e-6:
    print("\n✅ SUCCESS: Solver converged to target tolerance!")
    print(f"\nCalculating welfare...")
    _, results, d_trade = balanced_trade_multisector(result.x, data, param)
    print(f"USA welfare change: {results[id_US_new, 0]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade:.2f}%")
else:
    print("\n⚠️ WARNING: Solver did not fully converge")
    print(f"Final error ({np.max(np.abs(ceq_final)):.2e}) exceeds target (1e-6)")
