"""
Multi-sector model with FIXED initial guess
The key fix: normalize ell_ik_h so that sum(ell_ik * ell_ik_h) = 1
"""
import numpy as np
import pandas as pd
import sys
import os
from scipy.optimize import fsolve, root

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def balanced_trade_eq(x, data, param):
    """Equilibrium equations matching MATLAB exactly"""
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape(N, 1, K)

    # 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Trade shares
    AUX0 = lambda_ji * (w_i_3D / (L_ik_3D ** psi)) ** (-eps) * (1 + t_ji) ** (-eps * phi_3D)
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    P_i_h = (E_i_h / w_i_h) ** (1 - phi) * np.prod(
        np.sum(AUX0, axis=0, keepdims=True) ** (-beta_i[0:1, :, :] / eps[0:1, :, :]), axis=2
    ).reshape(-1)

    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    tau_i = tariff_rev / Y_i_new
    tau_i_h = 1.0 / (1 - tau_i)

    # ERR1: Sectoral income balance
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N * K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)

    # ERR2: Total income balance
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    # ERR4: Sectoral shares sum to 1
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Results
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)

    X_ji = lambda_ji * beta_i * np.tile(E_i.reshape(1, -1, 1), (N, 1, K))
    D_i = np.sum(np.sum(X_ji, axis=0), axis=1) - np.sum(np.sum(X_ji, axis=1), axis=1)
    D_i_new = np.sum(np.sum(X_ji_new, axis=0), axis=1) - np.sum(np.sum(X_ji_new, axis=1), axis=1)

    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=1) / Y_i_new) /
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=1) / Y_i) - 1)
    d_import = 100 * ((np.sum(np.sum(X_ji_new, axis=2) * (1 - np.eye(N)), axis=0) / Y_i_new) /
                      (np.sum(np.sum(X_ji, axis=2) * (1 - np.eye(N)), axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / (np.abs(D_i) + 1e-10))

    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, d_employment, d_CPI, tariff_rev / E_i])

    trade = X_ji * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def main():
    print("=" * 80)
    print("Multi-Sector Model with FIXED Initial Guess")
    print("=" * 80)

    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(base_path, 'python_output')

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
    print(f"N = {N} countries")

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
    print(f"US index = {id_US_new}")

    t_ji = np.zeros((N, N, K))
    t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
    t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
    t_ji[id_US_new, id_US_new, :K-1] = 0

    # Load parameters
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Calculate baseline values
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

    # Trade elasticities
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

    # =============================================
    # KEY FIX: Compute correct initial guess for ell_ik_h
    # =============================================
    sum_ell_ik = np.sum(ell_ik, axis=2, keepdims=True)  # Shape (N, 1, 1)
    ell_ik_h_init = 1.0 / sum_ell_ik  # This makes sum(ell_ik * ell_ik_h) = 1

    print(f"\nsum(ell_ik) range: [{sum_ell_ik.min():.4f}, {sum_ell_ik.max():.4f}]")
    print(f"ell_ik_h_init range: [{ell_ik_h_init.min():.4f}, {ell_ik_h_init.max():.4f}]")

    # Verify: sum(ell_ik * ell_ik_h_init) should be 1
    check = np.sum(ell_ik * ell_ik_h_init, axis=2)
    print(f"sum(ell_ik * ell_ik_h_init) range: [{check.min():.6f}, {check.max():.6f}]")

    # Create improved initial guess
    # ell_ik_h needs to be (N, 1, K) but we need N*K values
    # Broadcast ell_ik_h_init (N, 1, 1) to (N, 1, K) then flatten
    ell_ik_h_full = np.tile(ell_ik_h_init, (1, 1, K))  # Now (N, 1, K)

    x0 = np.concatenate([
        np.ones(N),  # w_i_h
        np.ones(N),  # E_i_h
        np.ones(N),  # L_i_h
        ell_ik_h_full.flatten()  # ell_ik_h (normalized, N*K values)
    ])

    print(f"\nInitial guess size: {len(x0)}")

    # =============================================
    # Scenario 1: No retaliation
    # =============================================
    print("\n" + "-" * 80)
    print("Scenario 1: USTR tariffs (no retaliation)")
    print("-" * 80)

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param = [eps_3D, kappa, psi, phi]

    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param)
        return ceq

    # Check initial error with fixed guess
    ceq_init = syst(x0)
    print(f"Initial error with FIXED guess: {np.max(np.abs(ceq_init)):.2e}")

    # Check initial error with old guess (all ones)
    x0_old = np.ones(3*N + N*K)
    ceq_old = syst(x0_old)
    print(f"Initial error with OLD guess (all ones): {np.max(np.abs(ceq_old)):.2e}")

    print(f"\nImprovement: {np.max(np.abs(ceq_old)) / np.max(np.abs(ceq_init)):.1f}x better")

    # Solve
    print("\nSolving with fsolve...")
    x_sol = fsolve(syst, x0, full_output=True, maxfev=500000)
    x_fsolve = x_sol[0]
    info = x_sol[1]

    ceq_final, results_multi[:, :, 0], d_trade_multi[0] = balanced_trade_eq(x_fsolve, data, param)
    final_error = np.max(np.abs(ceq_final))
    print(f"Final error: {final_error:.2e}")

    if final_error < 1.0:
        d_employment_multi[0] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)
        print(f"\n✅ Scenario 1 CONVERGED!")
        print(f"  USA welfare change: {results_multi[id_US_new, 0, 0]:.2f}%")
        print(f"  Global trade-to-GDP change: {d_trade_multi[0]:.2f}%")
        print(f"  Global employment change: {d_employment_multi[0]:.4f}%")

        # Target values from MATLAB
        print(f"\n  Target values (from MATLAB):")
        print(f"    USA welfare: 0.60%")
        print(f"    Global trade: -5.5%")
    else:
        print(f"\n⚠️ Scenario 1 did not fully converge (error={final_error:.2e})")

    # =============================================
    # Scenario 2: Reciprocal retaliation
    # =============================================
    print("\n" + "-" * 80)
    print("Scenario 2: Reciprocal retaliation")
    print("-" * 80)

    # Add reciprocal tariffs
    for k in range(K - 1):
        t_ji[id_US_new, :, k] = t_ji[:, id_US_new, k]
    t_ji[id_US_new, id_US_new, :] = 0

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]

    # Use solution from scenario 1 as starting point
    x_sol2 = fsolve(syst, x_fsolve, full_output=True, maxfev=500000)
    x_fsolve2 = x_sol2[0]

    ceq_final2, results_multi[:, :, 1], d_trade_multi[1] = balanced_trade_eq(x_fsolve2, data, param)
    final_error2 = np.max(np.abs(ceq_final2))
    print(f"Final error: {final_error2:.2e}")

    if final_error2 < 1.0:
        d_employment_multi[1] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)
        print(f"\n✅ Scenario 2 CONVERGED!")
        print(f"  USA welfare change: {results_multi[id_US_new, 0, 1]:.2f}%")
        print(f"  Global trade-to-GDP change: {d_trade_multi[1]:.2f}%")
        print(f"  Global employment change: {d_employment_multi[1]:.4f}%")

        print(f"\n  Target values (from MATLAB):")
        print(f"    USA welfare: -1.02%")
        print(f"    Global trade: -6.9%")
    else:
        print(f"\n⚠️ Scenario 2 did not fully converge (error={final_error2:.2e})")

    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    np.savez(os.path.join(output_dir, 'multisector_fixed_results.npz'),
             results_multi=results_multi,
             d_trade_multi=d_trade_multi,
             d_employment_multi=d_employment_multi,
             x_sol1=x_fsolve,
             x_sol2=x_fsolve2)
    print(f"Results saved to: multisector_fixed_results.npz")

    print("\n" + "=" * 80)
    print("Multi-Sector Fixed Model Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
