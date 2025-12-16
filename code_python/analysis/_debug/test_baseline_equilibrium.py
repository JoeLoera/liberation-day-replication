"""
Test: Verify equilibrium equations at baseline (zero tariffs, x = all ones)
At baseline, all errors should be approximately zero.
"""
import numpy as np
import pandas as pd
import os

def test_baseline():
    """Test that equilibrium errors are zero at baseline"""
    print("=" * 80)
    print("Testing Baseline Equilibrium (zero tariffs, x = all ones)")
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

    # Remove problematic countries
    problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
    ID = np.where(problematic_id == 1)[0]
    idx = np.setdiff1d(np.arange(N), ID)
    N = len(idx)
    print(f"N = {N} countries after filtering")

    X_new = np.zeros((N, N, K))
    for k in range(K):
        X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
    X_ji = X_new

    # ZERO TARIFFS for baseline test
    t_ji = np.zeros((N, N, K))

    # Load parameters
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Calculate baseline values (matching MATLAB lines 47-58)
    E_i = np.sum(np.sum(X_ji, axis=0), axis=1)  # Total expenditure
    Y_i = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
          nu * np.sum(np.sum(X_ji, axis=0), axis=1)  # Total income
    T_i = E_i - Y_i  # Trade deficit

    print(f"\nBaseline check:")
    print(f"  sum(E_i) = {np.sum(E_i):.2e}")
    print(f"  sum(Y_i) = {np.sum(Y_i):.2e}")
    print(f"  sum(T_i) = {np.sum(T_i):.2e} (should be ~0)")

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

    print(f"\nParameters:")
    print(f"  kappa = {kappa}")
    print(f"  psi = {psi}")
    print(f"  eps = {eps}")

    # At baseline: x = all ones
    x = np.ones(3*N + N*K)

    w_i_h = x[:N]
    E_i_h = x[N:2*N]
    L_i_h = x[2*N:3*N]
    ell_ik_h = x[3*N:].reshape(N, 1, K)

    print(f"\nInitial guess:")
    print(f"  w_i_h = all ones")
    print(f"  E_i_h = all ones")
    print(f"  L_i_h = all ones")
    print(f"  ell_ik_h = all ones")

    # Compute equilibrium (matching MATLAB lines 113-160)

    # Line 113-114: Construct 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))

    # Line 116: phi_3D
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Line 118: Trade shares
    AUX0 = lambda_ji * (w_i_3D / (L_ik_3D ** psi)) ** (-eps_3D) * (1 + t_ji) ** (-eps_3D * phi_3D)
    AUX1 = np.tile(np.sum(AUX0, axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1

    print(f"\nTrade share computation:")
    print(f"  At baseline (t_ji=0, x=1): lambda_ji_new should equal lambda_ji")
    print(f"  max|lambda_ji_new - lambda_ji| = {np.max(np.abs(lambda_ji_new - lambda_ji)):.2e}")

    # Line 122-124: Income and expenditure
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    print(f"\nIncome/Expenditure:")
    print(f"  Y_i_new = Y_i (since Y_i_h = 1)")
    print(f"  E_i_new = E_i (since E_i_h = 1)")

    # Line 127: Price index
    P_i_h = (E_i_h / w_i_h) ** (1 - phi) * np.prod(
        np.sum(AUX0, axis=0, keepdims=True) ** (-beta_i[0:1, :, :] / eps_3D[0:1, :, :]), axis=2
    ).reshape(-1)

    print(f"\nPrice index P_i_h:")
    print(f"  At baseline: P_i_h should be close to 1")
    print(f"  max|P_i_h - 1| = {np.max(np.abs(P_i_h - 1)):.2e}")
    print(f"  mean(P_i_h) = {np.mean(P_i_h):.4f}")

    # Line 129-130: Trade flows and tariff revenue
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = np.sum(np.sum(t_ji * X_ji_new, axis=2), axis=0)

    print(f"\nTariff revenue (should be 0 with zero tariffs):")
    print(f"  max(tariff_rev) = {np.max(tariff_rev):.2e}")

    # Line 132-134: Tax rates
    tau_i = tariff_rev / (Y_i_new + 1e-10)
    tau_i_h = 1.0 / (1 - tau_i)

    print(f"\nTax rates:")
    print(f"  max(tau_i) = {np.max(tau_i):.2e} (should be 0)")
    print(f"  tau_i_h should be 1: max|tau_i_h - 1| = {np.max(np.abs(tau_i_h - 1)):.2e}")

    # Line 138-143: ERR1 - Sectoral income balance
    nu_3D = np.tile(nu.reshape(1, -1, 1), (N, 1, K))
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik_baseline = ell_ik * np.tile(Y_i.reshape(-1, 1, 1), (1, 1, K))
    Y_ik_cf = np.sum((1 - nu_3D) * X_ji_new, axis=1, keepdims=True) + \
              np.transpose(np.sum(nu_3D * X_ji_new, axis=0, keepdims=True), (1, 0, 2))

    ERR1 = (Y_ik_cf - Y_ik_baseline * Y_ik_h).reshape(N * K)
    ERR1[N-1] = np.mean((P_i_h - 1) * E_i)  # Replace one equation

    print(f"\nERR1 (Sectoral income balance):")
    print(f"  max|ERR1| = {np.max(np.abs(ERR1)):.2e}")
    print(f"  mean|ERR1| = {np.mean(np.abs(ERR1)):.2e}")

    # Line 148-151: ERR2 - Total income balance
    X_global = np.sum(Y_i)
    X_global_new = np.sum(Y_i_new)
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    print(f"\nERR2 (Total income balance):")
    print(f"  max|ERR2| = {np.max(np.abs(ERR2)):.2e}")
    print(f"  mean|ERR2| = {np.mean(np.abs(ERR2)):.2e}")

    # Detailed breakdown of ERR2 components
    print(f"\n  ERR2 breakdown (should sum to 0):")
    print(f"    tariff_rev: {np.sum(tariff_rev):.2e}")
    print(f"    w*L*Y: {np.sum(w_i_h * L_i_h * Y_i):.2e}")
    print(f"    T*(X_new/X): {np.sum(T_i * (X_global_new / X_global)):.2e}")
    print(f"    E_new: {np.sum(E_i_new):.2e}")
    print(f"    Total: {np.sum(tariff_rev) + np.sum(w_i_h * L_i_h * Y_i) + np.sum(T_i * (X_global_new / X_global)) - np.sum(E_i_new):.2e}")

    # Line 155: ERR3 - Labor supply
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa

    print(f"\nERR3 (Labor supply):")
    print(f"  max|ERR3| = {np.max(np.abs(ERR3)):.2e}")
    print(f"  mean|ERR3| = {np.mean(np.abs(ERR3)):.2e}")

    # Line 158: ERR4 - Sectoral shares sum to 1
    ERR4 = 100 * (np.sum(ell_ik * ell_ik_h, axis=2).reshape(N) - 1)

    print(f"\nERR4 (Sectoral shares sum to 1):")
    print(f"  max|ERR4| = {np.max(np.abs(ERR4)):.2e}")
    print(f"  mean|ERR4| = {np.mean(np.abs(ERR4)):.2e}")

    # Total error
    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])
    total_error = np.max(np.abs(ceq))

    print(f"\n" + "=" * 80)
    print(f"TOTAL EQUILIBRIUM ERROR: {total_error:.2e}")
    print(f"=" * 80)

    if total_error < 1e-6:
        print("✅ PASS: Baseline equilibrium is satisfied!")
    else:
        print("❌ FAIL: Baseline equilibrium has errors")
        print("\nThis indicates a bug in the equation implementation.")

        # Find which error block is largest
        err_blocks = [
            ("ERR1", np.max(np.abs(ERR1))),
            ("ERR2", np.max(np.abs(ERR2))),
            ("ERR3", np.max(np.abs(ERR3))),
            ("ERR4", np.max(np.abs(ERR4)))
        ]
        err_blocks.sort(key=lambda x: x[1], reverse=True)
        print(f"\nError ranking:")
        for name, err in err_blocks:
            print(f"  {name}: {err:.2e}")

    return total_error


if __name__ == '__main__':
    test_baseline()
