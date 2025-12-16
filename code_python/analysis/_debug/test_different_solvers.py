"""
Try different solvers to see if any find the MATLAB solution
Target: d_trade = -5.5%, USA welfare = 0.60%
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve, root

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

# Load tariffs
tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
tariff_data_full = pd.read_csv(tariff_path)
new_ustariff = tariff_data_full.values[idx, :]

id_US = 185 - 1
id_US_new = np.where(idx == id_US + 1)[0][0]

t_ji = np.zeros((N, N, K))
t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, K-1))
t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
t_ji[id_US_new, id_US_new, :K-1] = 0

# Load parameters
phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
phi = phi_data['phi'].values[idx]
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

# Calculate baseline values
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

# Trade elasticities
Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

def balanced_trade_eq(x):
    """Equilibrium equations with F-order reshape"""
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

def get_results(x):
    """Compute results from solution"""
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

    tau_i = tariff_rev / Y_i_new
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji_baseline = lambda_ji * beta_i * E_i.reshape(1, -1, 1)
    trade_baseline = X_ji_baseline * (1 - np.eye(N)).reshape(N, N, 1)
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N)).reshape(N, N, 1)
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade_baseline)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return {
        'd_trade': d_trade,
        'd_welfare_US': 100 * (W_i_h[id_US_new] - 1),
        'd_employment': 100 * np.sum((L_i_h - 1) * Y_i) / np.sum(Y_i)
    }

# Try different solvers
x0 = np.ones(3*N + N*K)

print("\n" + "=" * 60)
print("Trying different solvers")
print("=" * 60)
print("Target: d_trade = -5.5%, USA welfare = 0.60%")

methods = [
    ('fsolve (default)', lambda: fsolve(balanced_trade_eq, x0, full_output=True)),
    ('root/hybr', lambda: root(balanced_trade_eq, x0, method='hybr')),
    ('root/lm', lambda: root(balanced_trade_eq, x0, method='lm')),
    ('root/broyden1', lambda: root(balanced_trade_eq, x0, method='broyden1', options={'maxiter': 10000})),
    ('root/krylov', lambda: root(balanced_trade_eq, x0, method='krylov', options={'maxiter': 10000})),
]

for name, solver_func in methods:
    print(f"\n{name}:")
    try:
        result = solver_func()
        if name.startswith('fsolve'):
            x_sol = result[0]
            info = result[1]
        else:
            x_sol = result.x

        ceq = balanced_trade_eq(x_sol)
        error = np.max(np.abs(ceq))

        if error < 10:  # Reasonable convergence
            results = get_results(x_sol)
            print(f"  Error: {error:.2e}")
            print(f"  d_trade: {results['d_trade']:.2f}%")
            print(f"  USA welfare: {results['d_welfare_US']:.2f}%")
        else:
            print(f"  Did not converge (error={error:.2e})")
    except Exception as e:
        print(f"  Failed: {str(e)[:50]}")

# Try with random initial guesses
print("\n" + "=" * 60)
print("Trying random initial guesses with fsolve")
print("=" * 60)

np.random.seed(42)
for i in range(5):
    # Random perturbation around 1
    x0_random = 1 + 0.1 * np.random.randn(3*N + N*K)
    x0_random = np.maximum(0.1, x0_random)  # Keep positive

    try:
        x_sol = fsolve(balanced_trade_eq, x0_random)
        ceq = balanced_trade_eq(x_sol)
        error = np.max(np.abs(ceq))

        if error < 10:
            results = get_results(x_sol)
            print(f"\nRandom seed {i}: Error={error:.2e}, d_trade={results['d_trade']:.2f}%, welfare={results['d_welfare_US']:.2f}%")
    except:
        pass

print("\n" + "=" * 60)
print("Trying initial guess with L_i_h < 1 (expecting employment to fall)")
print("=" * 60)

# Initial guess where L_i_h starts below 1
x0_low_L = np.ones(3*N + N*K)
x0_low_L[2*N:3*N] = 0.95  # Start with L_i_h = 0.95

x_sol = fsolve(balanced_trade_eq, x0_low_L)
ceq = balanced_trade_eq(x_sol)
error = np.max(np.abs(ceq))
results = get_results(x_sol)
print(f"Error: {error:.2e}")
print(f"d_trade: {results['d_trade']:.2f}%")
print(f"USA welfare: {results['d_welfare_US']:.2f}%")
print(f"Global employment: {results['d_employment']:.2f}%")
