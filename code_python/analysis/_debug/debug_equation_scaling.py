"""
Debug equation scaling and condition number
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve
from scipy.linalg import svd

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
id_US_new = np.where(idx == 185)[0][0]

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
print(f"Equations: {N*K + 3*N}")

# Check magnitudes
print("\nData magnitudes:")
print(f"  X_ji: [{X_ji.min():.2e}, {X_ji.max():.2e}], mean={X_ji.mean():.2e}")
print(f"  E_i: [{E_i.min():.2e}, {E_i.max():.2e}], mean={E_i.mean():.2e}")
print(f"  Y_i: [{Y_i.min():.2e}, {Y_i.max():.2e}], mean={Y_i.mean():.2e}")
print(f"  T: [{T.min():.2e}, {T.max():.2e}], mean={T.mean():.2e}")

def balanced_trade_eq(x, scale_y=1.0):
    """Equilibrium equations with optional scaling"""
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

    # SCALE ERR1 by dividing by baseline income
    ERR1_raw = (Y_ik_cf - Y_ik_baseline * Y_ik_h).reshape(N*K, order='F')
    ERR1 = ERR1_raw / (scale_y * np.tile(Y_i, K) + 1e-10)  # Scale by baseline income
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i) / (scale_y * np.mean(E_i))

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)

    # SCALE ERR2 by dividing by baseline expenditure
    ERR2_raw = tariff_rev + (w_i_h * L_i_h * Y_i) + T * (X_global_new / X_global) - E_i_new
    ERR2 = ERR2_raw / (scale_y * E_i + 1e-10)

    tau_i = tariff_rev / Y_i_new
    tau_i_h = 1.0 / (1 - tau_i)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    return np.concatenate([ERR1, ERR2, ERR3, ERR4])

# Check equation magnitudes at x = 1
x0 = np.ones(3*N + N*K)

print("\n" + "=" * 60)
print("Equation magnitudes at x = all ones (unscaled)")
print("=" * 60)
ceq = balanced_trade_eq(x0, scale_y=1.0)
print(f"ERR1: [{ceq[:N*K].min():.2e}, {ceq[:N*K].max():.2e}], mean abs = {np.abs(ceq[:N*K]).mean():.2e}")
print(f"ERR2: [{ceq[N*K:N*K+N].min():.2e}, {ceq[N*K:N*K+N].max():.2e}], mean abs = {np.abs(ceq[N*K:N*K+N]).mean():.2e}")
print(f"ERR3: [{ceq[N*K+N:N*K+2*N].min():.2e}, {ceq[N*K+N:N*K+2*N].max():.2e}], mean abs = {np.abs(ceq[N*K+N:N*K+2*N]).mean():.2e}")
print(f"ERR4: [{ceq[N*K+2*N:].min():.2e}, {ceq[N*K+2*N:].max():.2e}], mean abs = {np.abs(ceq[N*K+2*N:]).mean():.2e}")

# Solve with scaling
print("\n" + "=" * 60)
print("Solving with scaled equations")
print("=" * 60)

def syst_scaled(x):
    return balanced_trade_eq(x, scale_y=1.0)

x_sol = fsolve(syst_scaled, x0, full_output=True)
x_fsolve = x_sol[0]
info = x_sol[1]

ceq_final = syst_scaled(x_fsolve)
print(f"Final max error: {np.max(np.abs(ceq_final)):.2e}")
print(f"Function calls: {info['nfev']}")

# Check solution
w_i_h = np.abs(x_fsolve[:N])
E_i_h = np.abs(x_fsolve[N:2*N])
L_i_h = np.abs(x_fsolve[2*N:3*N])

print(f"\nSolution ranges:")
print(f"  w_i_h: [{w_i_h.min():.4f}, {w_i_h.max():.4f}]")
print(f"  E_i_h: [{E_i_h.min():.4f}, {E_i_h.max():.4f}]")
print(f"  L_i_h: [{L_i_h.min():.4f}, {L_i_h.max():.4f}]")
print(f"  USA w_i_h: {w_i_h[id_US_new]:.4f}")
print(f"  USA E_i_h: {E_i_h[id_US_new]:.4f}")
print(f"  USA L_i_h: {L_i_h[id_US_new]:.4f}")

# Compute d_trade
ell_ik_h = np.abs(x_fsolve[3*N:]).reshape((N, 1, K), order='F')
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
trade_baseline = X_ji_baseline * (1 - np.eye(N)).reshape(N, N, 1)
trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N)).reshape(N, N, 1)
d_trade = 100 * ((np.sum(trade_new) / np.sum(trade_baseline)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

print(f"\nd_trade: {d_trade:.2f}%")
print(f"Target: -5.5%")
