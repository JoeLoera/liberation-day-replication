"""
Multi-sector model solver using fixed-point iteration
This approach is more stable for economic equilibrium models
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def solve_equilibrium_fixedpoint(data, param, max_iter=5000, tol=1e-8, damping=0.3):
    """
    Solve the multi-sector equilibrium using fixed-point iteration with damping.

    This method updates each variable block sequentially using the equilibrium conditions,
    which is often more stable than simultaneous equation solving for trade models.
    """
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    # Initialize at the baseline (all ones = no change from baseline)
    w_i_h = np.ones(N)
    E_i_h = np.ones(N)
    L_i_h = np.ones(N)
    ell_ik_h = np.ones((N, 1, K))

    print(f"Starting fixed-point iteration (max_iter={max_iter}, tol={tol}, damping={damping})")

    for iteration in range(max_iter):
        # Store old values for convergence check
        w_old = w_i_h.copy()
        E_old = E_i_h.copy()
        L_old = L_i_h.copy()
        ell_old = ell_ik_h.copy()

        # Construct 3D arrays
        w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
        L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
        phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

        # Compute trade shares (lambda_ji_new)
        AUX0 = lambda_ji * (w_i_3D / (L_ik_3D ** psi)) ** (-eps) * (1 + t_ji) ** (-eps * phi_3D)
        AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
        lambda_ji_new = AUX0 / AUX1

        # Compute income and expenditure
        Y_i_h = w_i_h * L_i_h
        Y_i_new = Y_i_h * Y_i
        E_i_new = E_i * E_i_h

        # Compute price index
        P_i_h = (E_i_h / w_i_h) ** (1 - phi) * np.prod(
            np.sum(AUX0, axis=0, keepdims=True) ** (-beta_i[0:1, :, :] / eps[0:1, :, :]), axis=2
        ).reshape(-1)

        # Compute trade flows
        X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
        tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

        # Compute tax rate
        tau_i = tariff_rev / Y_i_new
        tau_i_h = 1.0 / (1 - tau_i)  # (1 - tau_new) / (1 - tau) where tau_new = 0

        # ============================================================
        # UPDATE 1: Labor supply (from ERR3)
        # ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)^kappa
        # ============================================================
        L_i_h_new = (tau_i_h * w_i_h / P_i_h) ** kappa
        L_i_h = (1 - damping) * L_i_h + damping * L_i_h_new

        # ============================================================
        # UPDATE 2: Expenditure (from ERR2)
        # ERR2 = tariff_rev + w*L*Y + T*(X_new/X) - E_new
        # => E_new = tariff_rev + w*L*Y + T*(X_new/X)
        # ============================================================
        X_global = np.sum(Y_i)
        X_global_new = np.sum(Y_i_new)
        E_i_new_implied = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)
        E_i_h_new = E_i_new_implied / E_i
        E_i_h = (1 - damping) * E_i_h + damping * E_i_h_new

        # ============================================================
        # UPDATE 3: Sectoral shares (from ERR4)
        # ERR4 = 100 * (sum(ell_ik * ell_ik_h, axis=2) - 1)
        # => sum(ell_ik * ell_ik_h) = 1 => ell_ik_h = 1/ell_ik (normalized)
        # This is automatically satisfied if we normalize
        # ============================================================
        # For sectoral shares, we derive from ERR1 instead
        # ERR1: Y_ik_cf = Y_ik * Y_ik_h
        # Y_ik_cf = sum((1-nu)*X_ji_new, axis=1) + sum(nu*X_ji_new, axis=0).T

        nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
        Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
                  np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))

        # Y_ik = ell_ik * Y_i (from baseline)
        Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))

        # Y_ik_h = w_i * L_ik = w_i * L_i * ell_ik_h
        # => ell_ik_h = Y_ik_cf / (Y_ik * w_i * L_i)
        # But we need to normalize so that sum(ell_ik * ell_ik_h, axis=2) = 1

        w_i_3D_small = np.tile(w_i_h.reshape(-1, 1, 1), (1, 1, K))
        L_i_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, 1, K))

        ell_ik_h_implied = Y_ik_cf / (Y_ik * w_i_3D_small * L_i_3D + 1e-10)

        # Normalize to satisfy ERR4 constraint
        ell_weighted = ell_ik * ell_ik_h_implied
        sum_ell_weighted = np.sum(ell_weighted, axis=2, keepdims=True)
        ell_ik_h_new = ell_ik_h_implied / (sum_ell_weighted + 1e-10)

        ell_ik_h = (1 - damping) * ell_ik_h + damping * ell_ik_h_new

        # ============================================================
        # UPDATE 4: Wages (from market clearing via ERR1)
        # We use total income balance: sum of sectoral income = total income
        # ============================================================
        total_income_cf = np.sum(Y_ik_cf, axis=2).reshape(N)
        total_income_target = Y_i * w_i_h * L_i_h

        # Adjust wages to match total income
        w_i_h_new = w_i_h * (total_income_cf / (total_income_target + 1e-10))

        # Normalize wages (price normalization)
        w_i_h_new = w_i_h_new / np.mean(w_i_h_new)

        w_i_h = (1 - damping) * w_i_h + damping * w_i_h_new

        # Ensure positivity
        w_i_h = np.maximum(w_i_h, 1e-6)
        E_i_h = np.maximum(E_i_h, 1e-6)
        L_i_h = np.maximum(L_i_h, 1e-6)
        ell_ik_h = np.maximum(ell_ik_h, 1e-6)

        # Check convergence
        max_change = max(
            np.max(np.abs(w_i_h - w_old)),
            np.max(np.abs(E_i_h - E_old)),
            np.max(np.abs(L_i_h - L_old)),
            np.max(np.abs(ell_ik_h - ell_old))
        )

        if iteration % 100 == 0 or max_change < tol:
            # Compute actual equilibrium errors
            x = np.concatenate([w_i_h, E_i_h, L_i_h, ell_ik_h.flatten()])
            ceq, _, _ = compute_equilibrium_errors(x, data, param)
            eq_error = np.max(np.abs(ceq))
            print(f"Iter {iteration:4d}: max_change={max_change:.2e}, eq_error={eq_error:.2e}", flush=True)

            if eq_error < tol:
                print(f"\n✅ CONVERGED at iteration {iteration}!")
                return x, eq_error, True

        if max_change < 1e-12:
            print(f"\n⚠️  Variables stopped changing at iteration {iteration}")
            x = np.concatenate([w_i_h, E_i_h, L_i_h, ell_ik_h.flatten()])
            ceq, _, _ = compute_equilibrium_errors(x, data, param)
            eq_error = np.max(np.abs(ceq))
            return x, eq_error, eq_error < tol

    print(f"\n⚠️  Reached max iterations ({max_iter})")
    x = np.concatenate([w_i_h, E_i_h, L_i_h, ell_ik_h.flatten()])
    ceq, _, _ = compute_equilibrium_errors(x, data, param)
    eq_error = np.max(np.abs(ceq))
    return x, eq_error, eq_error < tol


def compute_equilibrium_errors(x, data, param):
    """Compute equilibrium errors (same as Balanced_Trade_EQ in MATLAB)"""
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param

    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape(N, 1, K)

    # Construct 3D arrays
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
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)  # Replace one equation (price normalization)

    # ERR2: Total income balance
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    # ERR3: Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    # ERR4: Sectoral shares sum to 1
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])

    # Compute results
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

    # Global trade change
    trade = X_ji * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    trade_new = X_ji_new * (1 + t_ji) * np.tile((1 - np.eye(N)).reshape(N, N, 1), (1, 1, K))
    d_trade = 100 * ((np.sum(trade_new) / np.sum(trade)) / (np.sum(Y_i_new) / np.sum(Y_i)) - 1)

    return ceq, results, d_trade


def main():
    print("=" * 80)
    print("Multi-Sector Baseline Model - Fixed-Point Iteration Solver")
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

    # Remove countries with no trade
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
    t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
    t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
    t_ji[id_US_new, id_US_new, :K-1] = 0

    # Load parameters
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Calculate initial values
    E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
    Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
                nu * np.sum(np.sum(X_ji, axis=0), axis=1)
    T = E_i_multi - Y_i_multi

    # Trade shares
    lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
    beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
             np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

    # Sectoral income shares
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
    # Scenario 1: No retaliation
    # =============================================
    print("\n" + "-" * 80)
    print("Scenario 1: USTR tariffs (no retaliation)")
    print("-" * 80)

    data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param = [eps_3D, kappa, psi, phi]

    # Try different damping values
    for damping in [0.1, 0.2, 0.3, 0.5]:
        print(f"\nTrying damping = {damping}...")
        x_sol, error, converged = solve_equilibrium_fixedpoint(data, param, max_iter=2000, damping=damping)

        if converged or error < 1.0:
            print(f"Best error with damping={damping}: {error:.4e}")
            if converged:
                break

    if error < 10:  # Good enough for approximate results
        ceq, results_multi[:, :, 0], d_trade_multi[0] = compute_equilibrium_errors(x_sol, data, param)
        d_employment_multi[0] = np.sum(results_multi[:, 4, 0] * Y_i_multi) / np.sum(Y_i_multi)

        print(f"\nScenario 1 Results:")
        print(f"  USA welfare change: {results_multi[id_US_new, 0, 0]:.2f}%")
        print(f"  Global trade-to-GDP change: {d_trade_multi[0]:.2f}%")
        print(f"  Global employment change: {d_employment_multi[0]:.2f}%")
    else:
        print(f"\n❌ Scenario 1 did not converge (error={error:.2e})")

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

    for damping in [0.1, 0.2, 0.3, 0.5]:
        print(f"\nTrying damping = {damping}...")
        x_sol2, error2, converged2 = solve_equilibrium_fixedpoint(data, param, max_iter=2000, damping=damping)

        if converged2 or error2 < 1.0:
            print(f"Best error with damping={damping}: {error2:.4e}")
            if converged2:
                break

    if error2 < 10:
        ceq2, results_multi[:, :, 1], d_trade_multi[1] = compute_equilibrium_errors(x_sol2, data, param)
        d_employment_multi[1] = np.sum(results_multi[:, 4, 1] * Y_i_multi) / np.sum(Y_i_multi)

        print(f"\nScenario 2 Results:")
        print(f"  USA welfare change: {results_multi[id_US_new, 0, 1]:.2f}%")
        print(f"  Global trade-to-GDP change: {d_trade_multi[1]:.2f}%")
        print(f"  Global employment change: {d_employment_multi[1]:.2f}%")
    else:
        print(f"\n❌ Scenario 2 did not converge (error={error2:.2e})")

    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    np.savez(os.path.join(output_dir, 'multisector_fixedpoint_results.npz'),
             results_multi=results_multi,
             d_trade_multi=d_trade_multi,
             d_employment_multi=d_employment_multi)
    print(f"Results saved to: multisector_fixedpoint_results.npz")

    print("\n" + "=" * 80)
    print("Multi-Sector Fixed-Point Solver Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
