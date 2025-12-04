"""
Multi-sector baseline trade model.

This script replicates sub_multisector_baseline.m from the MATLAB replication package.
Converted from MATLAB to Python for:
"Making America Great Again? The Economic Impacts of Liberation Day Tariffs"
by Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os


def balanced_trade_eq_multi(x, data, param):
    """
    System of equations for balanced trade equilibrium with multiple sectors.

    Parameters:
    -----------
    x : np.ndarray
        Solution vector [w_i_h; E_i_h; L_i_h; ell_ik_h] (3N + NK x 1)
    data : dict
        Data dictionary
    param : dict
        Parameter dictionary

    Returns:
    --------
    ceq : np.ndarray
        Residuals
    results : np.ndarray
        Results matrix (N x 7)
    d_trade : float
        Change in global trade
    """
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data.values()
    eps, kappa, psi, phi = param.values()

    # Extract variables
    w_i_h = np.abs(x[0:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape(N, 1, K)

    # Construct 3D matrices
    wi_h_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    Lik_h_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * \
               np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(-1, 1, 1), (N, 1, K))

    # Construct new trade values
    AUX0 = lambda_ji * ((wi_h_3D / (Lik_h_3D ** psi)) ** (-eps)) * \
           ((1 + t_ji) ** (-eps * phi_3D))
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    # Price index
    P_i_h = ((E_i_h / w_i_h) ** (1 - phi)) * \
            np.prod(np.sum(AUX0, axis=0) ** (-beta_i[0, :, :] / eps[0, :, :]), axis=1)

    # New trade flows
    X_ji_new = lambda_ji_new * beta_i * \
               np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    # Tax adjustment
    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)

    # Equilibrium conditions
    # ERR1: Wage Income = Total Sales net of Taxes
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    Y_ik_h = wi_h_3D[:, 0, :] * Lik_h_3D[:, 0, :]
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=0) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=1, keepdims=True), (1, 0, 2))

    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N * K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)  # Replace one excess equation

    # ERR2: Total Income = Total Sales
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    # ERR4: Sectoral allocation
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2) - 1)
    ERR4 = ERR4.flatten()

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Calculate welfare
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    # Factual trade flows
    X_ji = lambda_ji * beta_i * np.tile(E_i.reshape(1, -1, 1), (N, 1, K))
    D_i = np.sum(np.sum(X_ji, axis=2), axis=1) - np.sum(np.sum(X_ji, axis=2), axis=0)
    D_i_new = np.sum(np.sum(X_ji_new, axis=2), axis=1) - np.sum(np.sum(X_ji_new, axis=2), axis=0)

    # Calculate changes
    d_welfare = 100 * (W_i_h - 1)
    eye_3d = np.tile(np.eye(N).reshape(N, N, 1), (1, 1, K))
    d_export = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=1) / Y_i_new) / \
                     (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=0) / Y_i_new) / \
                     (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import,
                              d_employment, d_CPI, tariff_rev / E_i])

    # Trade change
    trade = X_ji * (1 - eye_3d)
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / \
                    (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def run_multisector(base_results):
    """
    Run multi-sector baseline analysis.

    Parameters:
    -----------
    base_results : dict
        Results from main_baseline analysis

    Returns:
    --------
    dict : Dictionary with results
    """
    print("\n=== Running Multi-Sector Analysis ===")

    base_path = os.path.join(os.path.dirname(__file__), '..', '..')

    # Read multi-sector trade data
    data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
    data = pd.read_csv(data_path)
    X = data.iloc[:, 3].values  # 4th column (0-indexed = 3)

    N_orig = 194
    K = 4
    X_ji = X.reshape(N_orig, N_orig, K)

    # Read tariffs
    tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
    new_ustariff = pd.read_csv(tariff_path, header=None).values.flatten()
    id_US_orig = 185 - 1  # Convert to 0-indexed

    t_ji = np.zeros((N_orig, N_orig, K))
    t_ji[:, id_US_orig, 0:K-1] = np.tile(new_ustariff.reshape(-1, 1), (1, K-1))
    t_ji[:, id_US_orig, 0:K-1] = np.maximum(0.1, t_ji[:, id_US_orig, 0:K-1])
    t_ji[id_US_orig, id_US_orig, 0:K-1] = 0

    # Find problematic countries with no trade in any sector
    problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
    ID = np.where(problematic_id == 1)[0]
    idx = np.setdiff1d(np.arange(N_orig), ID)

    N = len(idx)
    X_new = np.zeros((N, N, K))
    t_new = np.zeros((N, N, K))
    for k in range(K):
        X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].squeeze()
        t_new[:, :, k] = t_ji[np.ix_(idx, idx, [k])].squeeze()

    X_ji = X_new
    t_ji = t_new

    id_US_new = np.where(idx == (id_US_orig))[0][0]

    # Get nu from base results
    nu = base_results['nu'][idx]

    # Calculate variables
    E_i_multi = np.sum(np.sum(X_ji, axis=1), axis=1)
    Y_i_multi = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=0).sum(axis=1) + \
                nu * np.sum(np.sum(X_ji, axis=1), axis=1)
    T = E_i_multi - Y_i_multi

    lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
    beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
             np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

    Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=0)
    Y_ik_f = np.tile(nu.reshape(1, 1, -1), (1, 1, K)) * np.sum(X_ji, axis=1, keepdims=True)
    Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
    ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))

    # Parameters
    kappa = 0.5
    psi = 0.67 / 4
    theta = 1 / psi

    # Get phi from base results
    phi = base_results['Phi'][0][idx]
    phi_avg = np.sum(base_results['Phi'][0] * base_results['Y_i']) / np.sum(base_results['Y_i'])

    eps = np.array([3.3, 3.8, 4.1]) / phi_avg
    eps = np.append(eps, 3)
    eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

    results_multi = np.zeros((N, 7, 2))
    d_trade = np.zeros(9)
    d_employment = np.zeros(9)

    # No Retaliation
    print("\nRunning multi-sector without retaliation...")
    data = {
        'N': N, 'K': K, 'E_i': E_i_multi, 'Y_i': Y_i_multi,
        'lambda_ji': lambda_ji, 'beta_i': beta_i, 'ell_ik': ell_ik,
        't_ji': t_ji, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps_3D, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N * K)])

    def syst(x):
        ceq, _, _ = balanced_trade_eq_multi(x, data, param)
        return ceq

    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    _, results_multi[:, :, 0], d_trade[7] = balanced_trade_eq_multi(x_fsolve, data, param)
    d_employment[7] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)
    print(f"  US welfare change: {results_multi[id_US_new, 0, 0]:.2f}%")

    # Reciprocal Retaliation
    print("\nRunning multi-sector with reciprocal retaliation...")
    for k in range(K-1):
        t_ji[id_US_new, :, k] = t_ji[:, id_US_new, k]
    t_ji[id_US_new, id_US_new, :] = 0

    data = {
        'N': N, 'K': K, 'E_i': E_i_multi, 'Y_i': Y_i_multi,
        'lambda_ji': lambda_ji, 'beta_i': beta_i, 'ell_ik': ell_ik,
        't_ji': t_ji, 'nu': nu, 'T_i': T
    }
    param = {'eps': eps_3D, 'kappa': kappa, 'psi': psi, 'phi': phi}

    x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=100000)
    _, results_multi[:, :, 1], d_trade[8] = balanced_trade_eq_multi(x_fsolve, data, param)
    d_employment[8] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)
    print(f"  US welfare change: {results_multi[id_US_new, 0, 1]:.2f}%")

    print("\n=== Multi-sector analysis completed ===")

    # Update trade and employment arrays from base results
    d_trade_full = base_results['d_trade']
    d_trade_full[7:9] = d_trade[7:9]

    d_employment_full = base_results['d_employment']
    d_employment_full[7:9] = d_employment[7:9]

    return {
        'results_multi': results_multi,
        'd_trade': d_trade_full,
        'd_employment': d_employment_full,
        'N_multi': N,
        'N': 194
    }


if __name__ == '__main__':
    # This would normally be called from main_baseline
    print("This script should be run from main_baseline.py")
    print("Run: python main_baseline.py")
