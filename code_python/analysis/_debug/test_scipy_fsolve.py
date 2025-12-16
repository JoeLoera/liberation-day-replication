"""
Test different scipy solvers to match MATLAB's fsolve behavior
MATLAB uses trust-region-dogleg by default (not Levenberg-Marquardt)
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve, root, least_squares

base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

# Load data (same setup as main code)
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

tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
new_ustariff = pd.read_csv(tariff_path).values[idx, :]
id_US_new = np.where(idx == 184)[0][0]

t_ji = np.zeros((N, N, K))
t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, K-1))
t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
t_ji[id_US_new, id_US_new, :K-1] = 0

phi = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
nu = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))['nu'].values[idx]

E_i = np.sum(np.sum(X_ji, axis=0), axis=1)
Y_i = np.sum((1 - nu).reshape(1, -1) * np.sum(X_ji, axis=2), axis=1) + \
      nu * np.sum(np.sum(X_ji, axis=0), axis=1)
T = E_i - Y_i

lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
         np.tile(E_i.reshape(1, -1, 1), (N, 1, K))

Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
ell_ik = Y_ik / np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))

Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_avg = np.sum(phi * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

print(f"N = {N}, K = {K}")
print(f"Variables: {3*N + N*K}")
print(f"Tariff mean (excl US): {t_ji[np.arange(N) != id_US_new, id_US_new, 0].mean():.4f}")

def balanced_trade_eq(x):
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K), order='F')

    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps_3D) * (1 + t_ji)**(-eps_3D * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    P_i_h = (E_i_h / w_i_h)**(1 - phi) * np.prod(
        np.sum(AUX0, axis=0, keepdims=True)**(-beta_i[0:1, :, :] / eps_3D[0:1, :, :]), axis=2
    ).reshape(-1)

    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik_baseline = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))

    ERR1 = (Y_ik_cf - Y_ik_baseline * Y_ik_h).reshape(N*K, order='F')
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T * (X_global_new / X_global) - E_i_new

    tau_i = tariff_rev / Y_i_new
    tau_i_h = 1.0 / (1 - tau_i)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    return np.concatenate([ERR1, ERR2, ERR3, ERR4])

def compute_d_trade(x):
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K), order='F')

    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps_3D) * (1 + t_ji)**(-eps_3D * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    lambda_ji_new = AUX0 / np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))

    Y_i_new = w_i_h * L_i_h * Y_i
    E_i_new = E_i * E_i_h
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)

    X_ji_baseline = lambda_ji * beta_i * E_i.reshape(1, -1, 1)
    trade = X_ji_baseline * (1 - np.eye(N)).reshape(N, N, 1)
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N)).reshape(N, N, 1)

    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)
    return d_trade

x0 = np.ones(3*N + N*K)

print("\n" + "=" * 60)
print("Testing different solvers")
print("=" * 60)

# Test 1: scipy.optimize.fsolve (modified Powell hybrid)
print("\n1. scipy.optimize.fsolve (modified Powell hybrid):")
try:
    x_sol, info, ier, msg = fsolve(balanced_trade_eq, x0, full_output=True)
    print(f"   Status: {msg}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(x_sol))):.2e}")
    print(f"   d_trade: {compute_d_trade(x_sol):.2f}%")
    print(f"   Function calls: {info['nfev']}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: root with method='hybr' (same as fsolve)
print("\n2. root method='hybr' (same algorithm as fsolve):")
try:
    sol = root(balanced_trade_eq, x0, method='hybr', tol=1e-10)
    print(f"   Success: {sol.success}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(sol.x))):.2e}")
    print(f"   d_trade: {compute_d_trade(sol.x):.2f}%")
    print(f"   Function calls: {sol.nfev}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: root with method='lm' (Levenberg-Marquardt)
print("\n3. root method='lm' (Levenberg-Marquardt):")
try:
    sol = root(balanced_trade_eq, x0, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000000})
    print(f"   Success: {sol.success}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(sol.x))):.2e}")
    print(f"   d_trade: {compute_d_trade(sol.x):.2f}%")
    print(f"   Function calls: {sol.nfev}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: least_squares with trust-region-reflective (closer to MATLAB)
print("\n4. least_squares with 'trf' (trust-region-reflective):")
try:
    sol = least_squares(balanced_trade_eq, x0, method='trf', ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    print(f"   Status: {sol.message}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(sol.x))):.2e}")
    print(f"   d_trade: {compute_d_trade(sol.x):.2f}%")
    print(f"   Function calls: {sol.nfev}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: least_squares with dogbox (dogleg-like)
print("\n5. least_squares with 'dogbox':")
try:
    sol = least_squares(balanced_trade_eq, x0, method='dogbox', ftol=1e-10, xtol=1e-10, max_nfev=1000000)
    print(f"   Status: {sol.message}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(sol.x))):.2e}")
    print(f"   d_trade: {compute_d_trade(sol.x):.2f}%")
    print(f"   Function calls: {sol.nfev}")
except Exception as e:
    print(f"   Error: {e}")

# Test 6: Try with a perturbed initial guess
print("\n6. Perturbed initial guess (simulating tariff effect):")
x0_perturbed = np.ones(3*N + N*K)
x0_perturbed[:N] = 1.0  # wages
x0_perturbed[N:2*N] = 0.98  # expenditure down
x0_perturbed[2*N:3*N] = 0.95  # labor down
try:
    sol = root(balanced_trade_eq, x0_perturbed, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000000})
    print(f"   Success: {sol.success}")
    print(f"   Max error: {np.max(np.abs(balanced_trade_eq(sol.x))):.2e}")
    print(f"   d_trade: {compute_d_trade(sol.x):.2f}%")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Checking tariff impact on trade shares at solution")
print("=" * 60)

# Get solution from best solver
sol = root(balanced_trade_eq, x0, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000000})
x_sol = sol.x

w_i_h = np.abs(x_sol[:N])
E_i_h = np.abs(x_sol[N:2*N])
L_i_h = np.abs(x_sol[2*N:3*N])
ell_ik_h = np.abs(x_sol[3*N:]).reshape((N, 1, K), order='F')

w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps_3D) * (1 + t_ji)**(-eps_3D * phi_3D)
AUX0 = lambda_ji * p_ij_h
lambda_ji_new = AUX0 / np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))

print(f"Trade share changes to US (sector 0):")
print(f"  lambda_ji[:, id_US, 0] (baseline): [{lambda_ji[:, id_US_new, 0].min():.6f}, {lambda_ji[:, id_US_new, 0].max():.6f}]")
print(f"  lambda_ji_new[:, id_US, 0]:        [{lambda_ji_new[:, id_US_new, 0].min():.6f}, {lambda_ji_new[:, id_US_new, 0].max():.6f}]")
print(f"  US domestic share (baseline): {lambda_ji[id_US_new, id_US_new, 0]:.6f}")
print(f"  US domestic share (new):      {lambda_ji_new[id_US_new, id_US_new, 0]:.6f}")

# Check if trade shares to US from foreign countries decreased
foreign = np.arange(N) != id_US_new
print(f"\nForeign share of US market:")
print(f"  Baseline: {np.sum(lambda_ji[foreign, id_US_new, 0]):.6f}")
print(f"  New:      {np.sum(lambda_ji_new[foreign, id_US_new, 0]):.6f}")
print(f"  Change:   {100*(np.sum(lambda_ji_new[foreign, id_US_new, 0])/np.sum(lambda_ji[foreign, id_US_new, 0]) - 1):.2f}%")
