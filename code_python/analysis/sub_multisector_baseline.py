"""
Multi-sector baseline model (K=4 sectors)
Generates results for Table 11 columns: "multi" (before & after retaliation)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def balanced_trade_multisector(x, data, param):
    """
    Multi-sector equilibrium system with K=4 sectors

    Variables (3N + NK):
    - w_i: wages (N)
    - E_i: expenditure (N)
    - L_i: labor (N)
    - ell_ik: sectoral labor shares (N*K)

    Parameters:
    - data: [N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i]
    - param: [eps, kappa, psi, phi]
    """
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    # Extract variables (use abs to avoid complex numbers)
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K))

    # Construct 3D arrays
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

    # New trade flows
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)

    # Tariff revenue
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    # Equilibrium equations
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))

    # Price index (needed for ERR1 replacement)
    P_i_h = (E_i_h / w_i_h)**(1 - phi) * np.prod(np.sum(AUX0, axis=0, keepdims=True)**(-beta_i[0:1, :, :] / eps[0:1, :, :]), axis=2).reshape(-1)

    # ERR1: Sectoral income balance (N*K equations)
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N*K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)  # Replace one redundant equation (matches MATLAB)

    # ERR2: Income = Sales + Transfers (N equations)
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply (N equations)
    tau_i = tariff_rev / Y_i_new  # Tariff revenue as fraction of NEW income
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Check for problematic values
    if np.any(tau_i > 0.99):
        print(f"WARNING: tau_i > 0.99 detected! Max tau_i = {np.max(tau_i):.4f}")
        print(f"Countries with high tau_i: {np.where(tau_i > 0.99)[0]}")

    # Labor supply equation: L_i = (tau_i * w_i / P_i)^kappa
    labor_term = tau_i_h * w_i_h / P_i_h
    if np.any(labor_term < 0):
        print(f"WARNING: Negative labor_term detected! Min = {np.min(labor_term):.4f}")

    ERR3 = L_i_h - labor_term**kappa

    # ERR4: Sectoral labor shares sum to 1 (N equations)
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Debug: Check equation magnitudes at first call
    if not hasattr(balanced_trade_multisector, 'debug_printed'):
        print(f"ERR1 shape: {ERR1.shape}, max: {np.max(np.abs(ERR1)):.2e}")
        print(f"ERR2 shape: {ERR2.shape}, max: {np.max(np.abs(ERR2)):.2e}")
        print(f"ERR3 shape: {ERR3.shape}, max: {np.max(np.abs(ERR3)):.2e}")
        print(f"ERR4 shape: {ERR4.shape}, max: {np.max(np.abs(ERR4)):.2e}")
        print(f"Total ceq shape: {ceq.shape}, max: {np.max(np.abs(ceq)):.2e}")
        print(f"E_i scale: min={np.min(E_i):.2e}, max={np.max(E_i):.2e}, mean={np.mean(E_i):.2e}")
        print(f"Y_i scale: min={np.min(Y_i):.2e}, max={np.max(Y_i):.2e}, mean={np.mean(Y_i):.2e}")
        print(f"Relative ERR2 (max ERR2 / max E_i): {np.max(np.abs(ERR2)) / np.max(E_i):.2e}")
        # Check which country has the max ERR2
        idx_max_err2 = np.argmax(np.abs(ERR2))
        print(f"Country with max ERR2: index {idx_max_err2}")
        print(f"  tariff_rev[{idx_max_err2}] = {tariff_rev[idx_max_err2]:.2e}")
        print(f"  w_i_h[{idx_max_err2}] * L_i_h[{idx_max_err2}] * Y_i[{idx_max_err2}] = {(w_i_h * L_i_h * Y_i)[idx_max_err2]:.2e}")
        print(f"  T_i[{idx_max_err2}] * (X_global_new / X_global) = {(T_i * (X_global_new / X_global))[idx_max_err2]:.2e}")
        print(f"  E_i_new[{idx_max_err2}] = {E_i_new[idx_max_err2]:.2e}")
        balanced_trade_multisector.debug_printed = True

    # Calculate results (baseline trade flows)
    # MATLAB: X_ji = lambda_ji.*beta_i.*repmat(E_i',N,1) where E_i is (N x 1)
    # repmat(E_i', N, 1) broadcasts E_i (as row vector) to (N x N)
    # Then MATLAB broadcasts (N x N) to (N x N x K) automatically
    X_ji = lambda_ji * beta_i * E_i.reshape(1, -1, 1)  # Natural broadcasting
    D_i = np.sum(np.sum(X_ji, axis=2), axis=0) - np.sum(np.sum(X_ji, axis=2), axis=1)
    D_i_new = np.sum(np.sum(X_ji_new, axis=2), axis=0) - np.sum(np.sum(X_ji_new, axis=2), axis=1)

    # Welfare calculation (matches MATLAB lines 162-163)
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=1) / Y_i_new) / \
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=0) / Y_i_new) / \
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, d_employment, d_CPI, tariff_rev / E_i])

    # Global trade change
    trade = X_ji * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))

    # Debug output
    if np.max(np.abs(ceq)) < 1e-6:  # Only print when converged
        trade_sum = np.sum(trade)
        trade_new_sum = np.sum(trade_new)
        Y_sum = np.sum(Y_i)
        Y_new_sum = np.sum(Y_i_new)
        trade_ratio = trade_new_sum / trade_sum if trade_sum > 0 else 0
        gdp_ratio = Y_new_sum / Y_sum if Y_sum > 0 else 0
        print(f"DEBUG: trade_sum={trade_sum:.2e}, trade_new_sum={trade_new_sum:.2e}, ratio={trade_ratio:.4f}")
        print(f"DEBUG: Y_sum={Y_sum:.2e}, Y_new_sum={Y_new_sum:.2e}, ratio={gdp_ratio:.4f}")
        print(f"DEBUG: trade_ratio/gdp_ratio={trade_ratio/gdp_ratio:.4f}")

    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def main():
    print("=" * 80)
    print("Multi-Sector Baseline Model (K=4 sectors)")
    print("=" * 80)

    # Set up base path
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    # Load sectoral trade data
    print("\nLoading sectoral trade data...")
    data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
    trade_data = pd.read_csv(data_path, header=None)
    X = trade_data.iloc[:, 3].values
    N = 194
    K = 4
    X_ji = X.reshape((N, N, K))

    # Remove countries with no trade FIRST
    problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
    ID = np.where(problematic_id == 1)[0]
    idx = np.setdiff1d(np.arange(N), ID)
    N = len(idx)

    X_new = np.zeros((N, N, K))
    for k in range(K):
        X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
    X_ji = X_new

    # Load and filter tariffs AFTER removing problematic countries
    print("Loading tariff data...")
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    tariff_data_full = pd.read_csv(tariff_path)
    new_ustariff_full = tariff_data_full.values
    # Filter tariffs to match filtered countries
    new_ustariff = new_ustariff_full[idx, :]

    id_US = 185 - 1  # Convert to 0-indexed (before filtering)
    id_US_new = np.where(idx == id_US + 1)[0][0]  # Find US index after filtering

    t_ji = np.zeros((N, N, K))
    t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
    t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
    t_ji[id_US_new, id_US_new, :K-1] = 0

    # Load parameters from baseline model
    print("Loading baseline parameters...")
    output_dir = os.path.join(base_path, 'python_output')
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Calculate initial values (MATLAB lines 47-49)
    # E_i = sum(sum(X_ji,1),3)' - total expenditure
    E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
    # Y_i = sum( repmat((1-nu)',N,1).*sum(X_ji,3) , 2) + nu.*sum(sum(X_ji,1),3)'
    Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
                nu * np.sum(np.sum(X_ji, axis=0), axis=1)
    # T = E_i - Y_i
    T = E_i_multi - Y_i_multi

    # Calculate trade share and expenditure share parameters (MATLAB lines 52-53)
    lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
    beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
             np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

    # Calculate sectoral income shares (MATLAB lines 55-58)
    # Y_ik_p = sum( repmat((1-nu)',[ N 1 K]).* X_ji , 2)
    Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
    # Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1)
    Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
    # Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3])
    Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
    # ell_ik = Y_ik./repmat( Y_i_multi, [1 1 K])
    ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

    # Trade elasticities
    Y_i_baseline = pd.read_csv(os.path.join(output_dir, 'Y_i_baseline.csv'))['Y_i'].values[idx]
    phi_baseline = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))['phi'].values[idx]
    phi_avg = np.sum(phi_baseline * Y_i_baseline) / np.sum(Y_i_baseline)
    eps = np.array([3.3, 3.8, 4.1]) / phi_avg
    eps = np.append(eps, 3.0)  # Services sector
    eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

    # Standard parameters
    kappa = 0.5
    psi = 0.67 / 4

    results_multi = np.zeros((N, 7, 2))
    d_trade_multi = np.zeros(2)
    d_employment_multi = np.zeros(2)

    # Scenario 1: No retaliation
    print("\n" + "-" * 80)
    print("Scenario 1: USTR tariffs (no retaliation)")
    print("-" * 80)

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param = [eps_3D, kappa, psi, phi]

    x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

    def syst(x):
        ceq, _, _ = balanced_trade_multisector(x, data, param)
        return ceq

    print("Solving equilibrium...")
    # Use fsolve with default algorithm (matches MATLAB's trust-region-dogleg)
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=50000)

    # Check convergence
    ceq_final = syst(x_fsolve)
    print(f"Max equilibrium error: {np.max(np.abs(ceq_final)):.2e}")

    _, results_multi[:, :, 0], d_trade_multi[0] = balanced_trade_multisector(x_fsolve, data, param)
    d_employment_multi[0] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)

    print(f"USA welfare change: {results_multi[id_US_new, 0, 0]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade_multi[0]:.2f}%")
    print(f"Global employment change: {d_employment_multi[0]:.2f}%")

    # Scenario 2: Reciprocal retaliation
    print("\n" + "-" * 80)
    print("Scenario 2: Reciprocal retaliation")
    print("-" * 80)

    for k in range(K-1):
        t_ji[id_US_new, :, k] = t_ji[:, id_US_new, k]
    t_ji[id_US_new, id_US_new, :] = 0

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]

    print("Solving equilibrium...")
    # Use fsolve with default algorithm (matches MATLAB's trust-region-dogleg)
    x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=50000)

    # Check convergence
    ceq_final = syst(x_fsolve)
    print(f"Max equilibrium error: {np.max(np.abs(ceq_final)):.2e}")

    _, results_multi[:, :, 1], d_trade_multi[1] = balanced_trade_multisector(x_fsolve, data, param)
    d_employment_multi[1] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)

    print(f"USA welfare change: {results_multi[id_US_new, 0, 1]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade_multi[1]:.2f}%")
    print(f"Global employment change: {d_employment_multi[1]:.2f}%")

    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    print("-" * 80)

    np.savez(os.path.join(output_dir, 'multisector_baseline_results.npz'),
             results_multi=results_multi,
             d_trade_multi=d_trade_multi,
             d_employment_multi=d_employment_multi,
             id_US=id_US_new)

    print("\nResults saved to: python_output/multisector_baseline_results.npz")
    print("  - d_trade_multi[0] (no retaliation): {:.2f}%".format(d_trade_multi[0]))
    print("  - d_trade_multi[1] (retaliation): {:.2f}%".format(d_trade_multi[1]))
    print("  - d_employment_multi[0] (no retaliation): {:.2f}%".format(d_employment_multi[0]))
    print("  - d_employment_multi[1] (retaliation): {:.2f}%".format(d_employment_multi[1]))

    print("\n" + "=" * 80)
    print("Multi-Sector Baseline Model Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
