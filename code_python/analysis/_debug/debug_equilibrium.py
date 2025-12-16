"""
Debug the equilibrium solution to understand why d_trade = 0
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

t_ji = np.zeros((N, N, K))
t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, K-1))
t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
t_ji[id_US_new, id_US_new, :K-1] = 0

print(f"\nTariff check:")
print(f"  Non-zero tariffs: {np.count_nonzero(t_ji)}")
print(f"  Max tariff: {t_ji.max():.4f}")
print(f"  t_ji[:, id_US_new, 0] non-zero: {np.count_nonzero(t_ji[:, id_US_new, 0])}")

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

# Verify ell_ik sums to 1
print(f"\nsum(ell_ik) check: [{np.sum(ell_ik, axis=2).min():.6f}, {np.sum(ell_ik, axis=2).max():.6f}]")

# Trade elasticities
Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
eps = np.array([3.3, 3.8, 4.1]) / phi_avg
eps = np.append(eps, 3.0)
eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

kappa = 0.5
psi = 0.67 / 4

print(f"\nParameters:")
print(f"  eps = {eps}")
print(f"  kappa = {kappa}")
print(f"  psi = {psi}")

# Initial guess
x0 = np.ones(3*N + N*K)

print("\n" + "=" * 60)
print("Testing equilibrium at x = all ones")
print("=" * 60)

# At x = all ones (baseline)
w_i_h = np.ones(N)
E_i_h = np.ones(N)
L_i_h = np.ones(N)
ell_ik_h = np.ones((N, 1, K))

# Compute key values at initial guess
w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

print(f"\nAt initial guess (x=1):")
print(f"  w_i_h = 1, E_i_h = 1, L_i_h = 1, ell_ik_h = 1")

# Trade shares
p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps_3D) * (1 + t_ji)**(-eps_3D * phi_3D)
AUX0 = lambda_ji * p_ij_h
AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
lambda_ji_new = AUX0 / AUX1

print(f"\nTrade share changes:")
print(f"  lambda_ji range: [{lambda_ji.min():.6f}, {lambda_ji.max():.6f}]")
print(f"  lambda_ji_new range: [{lambda_ji_new.min():.6f}, {lambda_ji_new.max():.6f}]")
print(f"  max|lambda_ji_new - lambda_ji|: {np.max(np.abs(lambda_ji_new - lambda_ji)):.6f}")

# Check if tariffs affect trade shares
# The key formula: lambda_ji_new = lambda_ji * (stuff) / sum(lambda_ji * (stuff))
# The (stuff) includes (1 + t_ji)**(-eps * phi)

# For countries exporting TO US (column id_US_new), they face tariffs
# So their (1 + t_ji)**(-eps*phi) term should be < 1
print(f"\nTariff effect on price term:")
tariff_term = (1 + t_ji[:, id_US_new, :])**(-eps_3D[:, id_US_new, :] * phi_3D[:, id_US_new, :])
print(f"  (1+t_ji)^(-eps*phi) for US imports: [{tariff_term.min():.6f}, {tariff_term.max():.6f}]")
print(f"  Number < 1: {np.sum(tariff_term < 1)}")
print(f"  Number = 1: {np.sum(tariff_term == 1)}")

# Check p_ij_h components
print(f"\nComponents of p_ij_h:")
wage_term = (w_i_3D / (L_ik_3D**psi))**(-eps_3D)
print(f"  (w/L^psi)^(-eps): [{wage_term.min():.6f}, {wage_term.max():.6f}]")
print(f"  At x=1, this should be 1.0: {np.allclose(wage_term, 1.0)}")

tariff_price_term = (1 + t_ji)**(-eps_3D * phi_3D)
print(f"  (1+t_ji)^(-eps*phi): [{tariff_price_term.min():.6f}, {tariff_price_term.max():.6f}]")
print(f"  Number != 1: {np.sum(np.abs(tariff_price_term - 1) > 1e-10)}")

# The issue: eps_3D * phi_3D
print(f"\neps * phi check:")
print(f"  eps_3D[:, :, 0]: {eps_3D[0, 0, 0]:.4f}")
print(f"  phi_3D[0, :5, 0]: {phi_3D[0, :5, 0]}")
print(f"  eps * phi for sector 0: [{(eps_3D[:,:,0] * phi_3D[:,:,0]).min():.4f}, {(eps_3D[:,:,0] * phi_3D[:,:,0]).max():.4f}]")

# Now let me trace through the full equilibrium
print("\n" + "=" * 60)
print("Solving equilibrium with tariffs")
print("=" * 60)

data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
param = [eps_3D, kappa, psi, phi]

def balanced_trade_eq_debug(x, data, param):
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K))

    # 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Price index
    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps) * (1 + t_ji)**(-eps * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    # Income and expenditure
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    # Trade flows
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)

    # Baseline trade flows for comparison
    X_ji_baseline = lambda_ji * beta_i * E_i.reshape(1, -1, 1)

    # Trade change (excluding diagonal)
    trade_baseline = X_ji_baseline * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))

    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade_baseline)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return {
        'w_i_h': w_i_h,
        'E_i_h': E_i_h,
        'L_i_h': L_i_h,
        'Y_i_new': Y_i_new,
        'E_i_new': E_i_new,
        'lambda_ji_new': lambda_ji_new,
        'X_ji_new': X_ji_new,
        'X_ji_baseline': X_ji_baseline,
        'trade_baseline': np.sum(trade_baseline),
        'trade_new': np.sum(trade_new),
        'd_trade': d_trade
    }

# Solve
from sub_multisector_baseline import balanced_trade_multisector

def syst(x):
    ceq, _, _ = balanced_trade_multisector(x, data, param)
    return ceq

print("Solving...")
sol = root(syst, x0, method='lm', options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000000})
x_sol = sol.x

print(f"Solver success: {sol.success}")
print(f"Max error: {np.max(np.abs(syst(x_sol))):.2e}")

# Debug the solution
debug_info = balanced_trade_eq_debug(x_sol, data, param)

print(f"\nSolution analysis:")
print(f"  w_i_h range: [{debug_info['w_i_h'].min():.6f}, {debug_info['w_i_h'].max():.6f}]")
print(f"  E_i_h range: [{debug_info['E_i_h'].min():.6f}, {debug_info['E_i_h'].max():.6f}]")
print(f"  L_i_h range: [{debug_info['L_i_h'].min():.6f}, {debug_info['L_i_h'].max():.6f}]")
print(f"  USA w_i_h: {debug_info['w_i_h'][id_US_new]:.6f}")
print(f"  USA E_i_h: {debug_info['E_i_h'][id_US_new]:.6f}")
print(f"  USA L_i_h: {debug_info['L_i_h'][id_US_new]:.6f}")

print(f"\nTrade analysis:")
print(f"  Baseline trade: {debug_info['trade_baseline']:.2e}")
print(f"  New trade: {debug_info['trade_new']:.2e}")
print(f"  Trade ratio: {debug_info['trade_new'] / debug_info['trade_baseline']:.6f}")
print(f"  Y ratio: {np.sum(debug_info['Y_i_new']) / np.sum(Y_i_multi):.6f}")
print(f"  d_trade: {debug_info['d_trade']:.2f}%")

# Check if the solution is trivial (x ≈ 1)
print(f"\nIs solution trivial (x ≈ 1)?")
x_deviation = np.abs(x_sol - 1)
print(f"  max|x - 1|: {x_deviation.max():.6f}")
print(f"  mean|x - 1|: {x_deviation.mean():.6f}")

# Check which components deviate most
w_dev = np.max(np.abs(x_sol[:N] - 1))
E_dev = np.max(np.abs(x_sol[N:2*N] - 1))
L_dev = np.max(np.abs(x_sol[2*N:3*N] - 1))
ell_dev = np.max(np.abs(x_sol[3*N:] - 1))
print(f"  max|w - 1|: {w_dev:.6f}")
print(f"  max|E - 1|: {E_dev:.6f}")
print(f"  max|L - 1|: {L_dev:.6f}")
print(f"  max|ell - 1|: {ell_dev:.6f}")
