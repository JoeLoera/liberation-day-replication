"""
Test initial equilibrium error at x0 = all ones
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sub_multisector_baseline import balanced_trade_multisector

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

# Load sectoral trade data
print("Loading sectoral trade data...")
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

# Load tariffs (same as in main script)
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

print(f"Tariffs loaded: max tariff = {np.max(t_ji):.2f}")

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

# Test TWO initial guesses
x0_bad = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])
x0_good = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)/K])

print(f"\nN = {N}, K = {K}")
print(f"Total variables: {len(x0_bad)}")
print(f"Expected: {3*N + N*K} = {3*N + N*K}")

# Evaluate equilibrium at x0 (BAD - unnormalized)
print("\n" + "="*80)
print("TEST 1: x0 = all ones (MATLAB default - unnormalized ell_ik)")
print("="*80)
ceq, _, _ = balanced_trade_multisector(x0_bad, data, param)

print(f"\nEquilibrium errors:")
print(f"  Total equations: {len(ceq)}")
print(f"  Max absolute error: {np.max(np.abs(ceq)):.2e}")
print(f"  Mean absolute error: {np.mean(np.abs(ceq)):.2e}")
print(f"  Median absolute error: {np.median(np.abs(ceq)):.2e}")
print(f"  Min absolute error: {np.min(np.abs(ceq)):.2e}")

# Break down by equation type
ERR1 = ceq[:N*K]
ERR2 = ceq[N*K:N*K+N]
ERR3 = ceq[N*K+N:N*K+2*N]
ERR4 = ceq[N*K+2*N:]

print(f"\nERR1 (sectoral income, {len(ERR1)} eqs):")
print(f"  Max: {np.max(np.abs(ERR1)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR1)):.2e}")

print(f"\nERR2 (total income, {len(ERR2)} eqs):")
print(f"  Max: {np.max(np.abs(ERR2)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR2)):.2e}")

print(f"\nERR3 (labor supply, {len(ERR3)} eqs):")
print(f"  Max: {np.max(np.abs(ERR3)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR3)):.2e}")

print(f"\nERR4 (sectoral shares, {len(ERR4)} eqs):")
print(f"  Max: {np.max(np.abs(ERR4)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR4)):.2e}")

# Now test GOOD initial guess
print("\n" + "="*80)
print("TEST 2: x0 with normalized ell_ik = 1/K (IMPROVED)")
print("="*80)
ceq, _, _ = balanced_trade_multisector(x0_good, data, param)

print(f"\nEquilibrium errors:")
print(f"  Total equations: {len(ceq)}")
print(f"  Max absolute error: {np.max(np.abs(ceq)):.2e}")
print(f"  Mean absolute error: {np.mean(np.abs(ceq)):.2e}")
print(f"  Median absolute error: {np.median(np.abs(ceq)):.2e}")
print(f"  Min absolute error: {np.min(np.abs(ceq)):.2e}")

# Break down by equation type
N_eq = len(idx)
ERR1 = ceq[:N*K]
ERR2 = ceq[N*K:N*K+N]
ERR3 = ceq[N*K+N:N*K+2*N]
ERR4 = ceq[N*K+2*N:]

print(f"\nERR1 (sectoral income, {len(ERR1)} eqs):")
print(f"  Max: {np.max(np.abs(ERR1)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR1)):.2e}")

print(f"\nERR2 (total income, {len(ERR2)} eqs):")
print(f"  Max: {np.max(np.abs(ERR2)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR2)):.2e}")

print(f"\nERR3 (labor supply, {len(ERR3)} eqs):")
print(f"  Max: {np.max(np.abs(ERR3)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR3)):.2e}")

print(f"\nERR4 (sectoral shares, {len(ERR4)} eqs):")
print(f"  Max: {np.max(np.abs(ERR4)):.2e}")
print(f"  Mean: {np.mean(np.abs(ERR4)):.2e}")
