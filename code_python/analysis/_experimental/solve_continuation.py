"""
Solve equilibrium using continuation/homotopy method
Start from zero tariffs (where solution = 1) and gradually increase to actual tariffs
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import root

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
new_ustariff_full = tariff_data_full.values
new_ustariff = new_ustariff_full[idx, :]

id_US = 185 - 1
id_US_new = np.where(idx == id_US + 1)[0][0]

# Full tariff matrix
t_ji_full = np.zeros((N, N, K))
t_ji_full[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, K-1))
t_ji_full[:, id_US_new, :K-1] = np.maximum(0.1, t_ji_full[:, id_US_new, :K-1])
t_ji_full[id_US_new, id_US_new, :K-1] = 0

print(f"Max tariff: {t_ji_full.max():.4f}")

# Load parameters
phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
phi = phi_data['phi'].values[idx]
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

# Calculate baseline values with CORRECT formula
E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
Y_i_multi = np.sum((1 - nu).reshape(1, -1) * np.sum(X_ji, axis=2), axis=1) + \
            nu * np.sum(np.sum(X_ji, axis=0), axis=1)
T = E_i_multi - Y_i_multi

lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
         np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

# Trade elasticities
Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

def balanced_trade_eq(x, t_ji, return_results=False):
    """Equilibrium equations"""
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K))

    # 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Trade shares
    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps_3D) * (1 + t_ji)**(-eps_3D * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    # Income and expenditure
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i_multi
    E_i_new = E_i_multi * E_i_h

    # Trade flows and tariff revenue
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    # Price index
    P_i_h = (E_i_h / w_i_h)**(1 - phi) * np.prod(
        np.sum(AUX0, axis=0, keepdims=True)**(-beta_i[0:1, :, :] / eps_3D[0:1, :, :]), axis=2
    ).reshape(-1)

    # ERR1: Sectoral income balance
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik_baseline = ell_ik * np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))
    ERR1 = (Y_ik_cf - Y_ik_baseline * Y_ik_h).reshape(N * K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i_multi)

    # ERR2: Total income balance
    X_global = np.sum(Y_i_multi)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i_multi) + T * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply
    tau_i = tariff_rev / Y_i_new
    tau_i_h = 1.0 / (1 - tau_i)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    # ERR4: Sectoral shares sum to 1
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    if not return_results:
        return ceq

    # Results
    delta_i = E_i_multi / (E_i_multi - kappa * (1 - tau_i) * Y_i_multi / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji_baseline = lambda_ji * beta_i * E_i_multi.reshape(1, -1, 1)
    trade_baseline = X_ji_baseline * (1 - np.eye(N)).reshape(N, N, 1)
    trade_new = X_ji_new * (1 + t_ji) * (1 - np.eye(N)).reshape(N, N, 1)
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade_baseline)) / (np.sum(Y_i_new) / np.sum(Y_i_multi)) - 1)

    d_welfare = 100 * (W_i_h - 1)
    d_employment = 100 * (L_i_h - 1)

    return {
        'ceq': ceq,
        'w_i_h': w_i_h,
        'E_i_h': E_i_h,
        'L_i_h': L_i_h,
        'd_trade': d_trade,
        'd_welfare_US': d_welfare[id_US_new],
        'd_employment': np.sum(d_employment * Y_i_multi) / np.sum(Y_i_multi)
    }

# Continuation method: gradually increase tariffs
print("\n" + "=" * 60)
print("Continuation method: gradually increasing tariffs")
print("=" * 60)

x_current = np.ones(3*N + N*K)
n_steps = 20
tariff_fractions = np.linspace(0, 1, n_steps + 1)

for i, frac in enumerate(tariff_fractions):
    t_ji_current = frac * t_ji_full

    def syst(x):
        return balanced_trade_eq(x, t_ji_current)

    sol = root(syst, x_current, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 100000})
    x_current = sol.x

    results = balanced_trade_eq(x_current, t_ji_current, return_results=True)
    error = np.max(np.abs(results['ceq']))

    if i % 5 == 0 or i == n_steps:
        print(f"\nStep {i}/{n_steps}: tariff fraction = {frac:.2f}")
        print(f"  Max error: {error:.2e}")
        print(f"  d_trade: {results['d_trade']:.2f}%")
        print(f"  USA welfare: {results['d_welfare_US']:.2f}%")
        print(f"  Global employment: {results['d_employment']:.4f}%")
        print(f"  max|x-1|: {np.max(np.abs(x_current - 1)):.4f}")

# Final solution at full tariffs
print("\n" + "=" * 60)
print("Final solution at full tariffs")
print("=" * 60)

results_final = balanced_trade_eq(x_current, t_ji_full, return_results=True)
print(f"Max error: {np.max(np.abs(results_final['ceq'])):.2e}")
print(f"d_trade: {results_final['d_trade']:.2f}%")
print(f"USA welfare: {results_final['d_welfare_US']:.2f}%")
print(f"Global employment: {results_final['d_employment']:.4f}%")

print("\nTarget values from MATLAB:")
print("  d_trade: -5.5%")
print("  USA welfare: 0.60%")

# Check if continuation found a different solution
print("\n" + "=" * 60)
print("Comparing continuation solution with direct solve")
print("=" * 60)

# Direct solve from x=1
def syst_full(x):
    return balanced_trade_eq(x, t_ji_full)

x_direct = np.ones(3*N + N*K)
sol_direct = root(syst_full, x_direct, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 100000})
results_direct = balanced_trade_eq(sol_direct.x, t_ji_full, return_results=True)

print(f"Direct solve (from x=1):")
print(f"  Max error: {np.max(np.abs(results_direct['ceq'])):.2e}")
print(f"  d_trade: {results_direct['d_trade']:.2f}%")
print(f"  USA welfare: {results_direct['d_welfare_US']:.2f}%")

print(f"\nContinuation solve:")
print(f"  Max error: {np.max(np.abs(results_final['ceq'])):.2e}")
print(f"  d_trade: {results_final['d_trade']:.2f}%")
print(f"  USA welfare: {results_final['d_welfare_US']:.2f}%")

# Check if solutions are different
diff = np.max(np.abs(x_current - sol_direct.x))
print(f"\nmax|x_continuation - x_direct|: {diff:.6f}")
