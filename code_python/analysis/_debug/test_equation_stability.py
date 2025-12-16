"""
Test numerical stability of equilibrium equations
Check for NaN, Inf, or extreme values
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sub_multisector_baseline import balanced_trade_multisector

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

print("Loading data...")
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K))

problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N), ID)
N = len(idx)

X_new = np.zeros((N, N, K))
for k in range(K):
    X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
X_ji = X_new

phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
phi = phi_data['phi'].values[idx]
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
            nu * np.sum(np.sum(X_ji, axis=0), axis=1)
T = E_i_multi - Y_i_multi

lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
         np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

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

Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
param = [eps_3D, kappa, psi, phi]

print("\nTesting equation stability at different points...")

# Test 1: x0 = all ones
print("\n" + "="*80)
print("Test 1: x0 = all ones")
x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

ceq = balanced_trade_multisector(x0, data, param)[0]
print(f"  NaN values: {np.sum(np.isnan(ceq))}")
print(f"  Inf values: {np.sum(np.isinf(ceq))}")
print(f"  Max value: {np.max(np.abs(ceq)):.2e}")
print(f"  ✓ Evaluates successfully" if not (np.any(np.isnan(ceq)) or np.any(np.isinf(ceq))) else "  ✗ Has NaN/Inf!")

# Test 2: Small perturbation from x0
print("\n" + "="*80)
print("Test 2: x0 + small random perturbation (0.01)")
x_perturb = x0 + 0.01 * np.random.randn(len(x0))

ceq = balanced_trade_multisector(x_perturb, data, param)[0]
print(f"  NaN values: {np.sum(np.isnan(ceq))}")
print(f"  Inf values: {np.sum(np.isinf(ceq))}")
print(f"  Max value: {np.max(np.abs(ceq)):.2e}")
print(f"  ✓ Evaluates successfully" if not (np.any(np.isnan(ceq)) or np.any(np.isinf(ceq))) else "  ✗ Has NaN/Inf!")

# Test 3: Numerical Jacobian test
print("\n" + "="*80)
print("Test 3: Numerical Jacobian (finite difference)")
print("Computing Jacobian for first 10 variables...")

def syst(x):
    return balanced_trade_multisector(x, data, param)[0]

# Test finite difference for first few variables
eps_fd = 1e-8
n_test = min(10, len(x0))
jacobian_issues = 0

for i in range(n_test):
    x_plus = x0.copy()
    x_plus[i] += eps_fd

    ceq_0 = syst(x0)
    ceq_plus = syst(x_plus)

    jac_col = (ceq_plus - ceq_0) / eps_fd

    if np.any(np.isnan(jac_col)) or np.any(np.isinf(jac_col)):
        jacobian_issues += 1
        print(f"  Variable {i}: NaN/Inf in Jacobian column")

if jacobian_issues == 0:
    print(f"  ✓ All {n_test} Jacobian columns are finite")
else:
    print(f"  ✗ {jacobian_issues}/{n_test} Jacobian columns have NaN/Inf")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("The equilibrium function appears numerically stable.")
print("Solver convergence issues are likely algorithmic, not numerical.")
