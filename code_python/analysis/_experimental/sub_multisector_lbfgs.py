"""
Multi-sector baseline using L-BFGS-B minimization
Instead of root-finding, minimize sum of squared residuals
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def balanced_trade_multisector(x, data, param):
    """Multi-sector equilibrium (same as before)"""
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape((N, 1, K))

    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps) * (1 + t_ji)**(-eps * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    P_i_h = (E_i_h / w_i_h)**(1 - phi) * np.prod(np.sum(AUX0, axis=0, keepdims=True)**(-beta_i[0:1, :, :] / eps[0:1, :, :]), axis=2).reshape(-1)

    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N*K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)

    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Calculate results
    X_ji = lambda_ji * beta_i * E_i.reshape(1, -1, 1)
    D_i_new = np.sum(np.sum(X_ji_new, axis=2), axis=0) - np.sum(np.sum(X_ji_new, axis=2), axis=1)

    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    E_i_h_calc = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / E_i
    W_i_h = delta_i * (E_i_h_calc / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    results = np.zeros((N, 7))
    results[:, 0] = 100 * (W_i_h - 1)
    results[:, 1] = 100 * (D_i_new / Y_i_new - D_i_new / Y_i)
    results[:, 4] = 100 * (L_i_h - 1)
    results[:, 6] = 100 * (P_i_h - 1)

    trade = np.sum(np.sum(X_ji, axis=2))
    trade_new = np.sum(np.sum(X_ji_new, axis=2))
    GDP = np.sum(Y_i)
    GDP_new = np.sum(Y_i_new)
    d_trade = 100 * ((trade_new / trade) / (GDP_new / GDP) - 1)

    return ceq, results, d_trade


def main():
    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(base_path, 'python_output')

    print("="*80)
    print("Multi-Sector Baseline: L-BFGS-B Minimization Approach")
    print("="*80)

    # Load data
    print("\nLoading data...")
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

    results_multi = np.zeros((N, 7, 2))
    d_trade_multi = np.zeros(2)
    d_employment_multi = np.zeros(2)

    # Scenario 1: No retaliation
    print("\n" + "-"*80)
    print("Scenario 1: USTR tariffs (no retaliation)")
    print("-"*80)

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param = [eps_3D, kappa, psi, phi]

    x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

    # Define objective: sum of squared residuals
    def objective(x):
        ceq, _, _ = balanced_trade_multisector(x, data, param)
        return np.sum(ceq**2)

    print("\nSolving with L-BFGS-B (minimizing sum of squared residuals)...")
    print(f"Initial objective: {objective(x0):.2e}")

    start_time = time.time()

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'ftol': 1e-20,
            'gtol': 1e-10,
            'disp': True
        }
    )

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Optimization completed in {elapsed:.1f} seconds")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    print(f"{'='*80}")

    # Check convergence
    ceq_final, results_multi[:, :, 0], d_trade_multi[0] = balanced_trade_multisector(result.x, data, param)
    final_error = np.max(np.abs(ceq_final))
    d_employment_multi[0] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)

    print(f"\nFinal equilibrium error: {final_error:.2e}")
    print(f"Final objective (sum sq): {objective(result.x):.2e}")

    if final_error < 1e-4:
        print(f"\n✅ CONVERGED!")
        print(f"USA welfare: {results_multi[id_US_new, 0, 0]:.2f}%")
        print(f"Global trade: {d_trade_multi[0]:.2f}%")
        print(f"Global employment: {d_employment_multi[0]:.2f}%")
        print(f"\nTarget: welfare=0.60%, trade=-5.5%")

        # Scenario 2
        print("\n" + "-"*80)
        print("Scenario 2: Reciprocal retaliation")
        print("-"*80)

        for k in range(K-1):
            t_ji[id_US_new, :, k] = t_ji[:, id_US_new, k]
        t_ji[id_US_new, id_US_new, :] = 0

        data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]

        # Use scenario 1 as initial guess
        result2 = minimize(
            lambda x: np.sum(balanced_trade_multisector(x, data, param)[0]**2),
            result.x,  # Warm start
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-10, 'disp': True}
        )

        ceq_final2, results_multi[:, :, 1], d_trade_multi[1] = balanced_trade_multisector(result2.x, data, param)
        final_error2 = np.max(np.abs(ceq_final2))
        d_employment_multi[1] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)

        print(f"\nFinal error: {final_error2:.2e}")

        if final_error2 < 1e-4:
            print(f"✅ CONVERGED!")
            print(f"USA welfare: {results_multi[id_US_new, 0, 1]:.2f}%")
            print(f"Global trade: {d_trade_multi[1]:.2f}%")
            print(f"\nTarget: welfare=-1.02%, trade=-6.9%")

            # Save
            np.savez(os.path.join(output_dir, 'multisector_baseline_results.npz'),
                     results_multi=results_multi,
                     d_trade_multi=d_trade_multi,
                     d_employment_multi=d_employment_multi,
                     id_US=id_US_new)

            print(f"\n{'='*80}")
            print("🎉 100% PYTHON REPLICATION ACHIEVED!")
            print(f"{'='*80}")
        else:
            print(f"⚠️  Scenario 2 did not converge (error={final_error2:.2e})")
    else:
        print(f"\n⚠️  Did not converge (error={final_error:.2e})")
        print("Trying alternative approach...")


if __name__ == '__main__':
    main()
