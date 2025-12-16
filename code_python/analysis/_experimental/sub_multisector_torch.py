"""
Multi-sector solver using PyTorch Adam optimizer
Adam uses momentum and adaptive learning rates - may escape flat regions
"""
import numpy as np
import pandas as pd
import sys
import os
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def balanced_trade_torch(x_tensor, data_np, param_np):
    """
    Equilibrium equations using PyTorch for automatic differentiation
    """
    N, K = data_np[0], data_np[1]
    E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = [
        torch.tensor(arr, dtype=torch.float64) for arr in data_np[2:]
    ]
    eps, kappa, psi, phi = [
        torch.tensor(arr, dtype=torch.float64) for arr in param_np
    ]

    # Extract and ensure positive
    w_i_h = torch.abs(x_tensor[:N])
    E_i_h = torch.abs(x_tensor[N:2*N])
    L_i_h = torch.abs(x_tensor[2*N:3*N])
    ell_ik_h = torch.abs(x_tensor[3*N:]).reshape(N, 1, K)

    # Construct 3D arrays
    w_i_3D = w_i_h.reshape(-1, 1, 1).expand(-1, N, K)
    L_ik_3D = L_i_h.reshape(-1, 1, 1).expand(-1, N, K) * ell_ik_h.expand(-1, N, -1)
    phi_3D = phi.reshape(1, -1, 1).expand(N, -1, K)

    # Price index
    p_ij_h = (w_i_3D / (L_ik_3D**psi))**(-eps) * (1 + t_ji)**(-eps * phi_3D)
    AUX0 = lambda_ji * p_ij_h
    AUX1 = AUX0.sum(dim=0, keepdim=True).expand(N, -1, -1)
    lambda_ji_new = AUX0 / AUX1

    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h

    X_ji_new = lambda_ji_new * beta_i * E_i_new.reshape(1, -1, 1).expand(N, -1, K) / (1 + t_ji)
    tariff_rev = (t_ji * X_ji_new).sum(dim=2).sum(dim=0)

    nu_3D = nu.reshape(1, -1, 1).expand(N, -1, K)
    P_i_h = (E_i_h / w_i_h)**(1 - phi) * (AUX0.sum(dim=0, keepdim=True)**(-beta_i[0:1, :, :] / eps[0:1, :, :])).prod(dim=2).reshape(-1)

    # Equilibrium equations
    Y_ik_h = w_i_3D[:, 0:1, :] * L_ik_3D[:, 0:1, :]
    Y_ik = ell_ik * Y_i.reshape(-1, 1, 1).expand(-1, -1, K)
    Y_ik_cf = ((1 - nu_3D) * X_ji_new).sum(dim=1, keepdim=True) + \
              (nu_3D * X_ji_new).sum(dim=0, keepdim=True).permute(1, 0, 2)
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N*K)
    ERR1[N-1] = ((P_i_h - 1) * E_i).mean()

    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new

    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h)**kappa

    ERR4 = 100 * ((ell_ik * ell_ik_h).sum(dim=2).reshape(N) - 1)

    ceq = torch.cat([ERR1, ERR2, ERR3, ERR4])
    return ceq


def solve_with_adam(data_np, param_np, max_iters=5000, lr=0.01):
    """
    Solve using PyTorch Adam optimizer
    """
    N, K = data_np[0], data_np[1]

    # Initial guess as PyTorch tensor with gradients
    x = torch.ones(3*N + N*K, dtype=torch.float64, requires_grad=True)

    # Adam optimizer
    optimizer = torch.optim.Adam([x], lr=lr)

    print(f"\nOptimizing with Adam (lr={lr}, max_iters={max_iters})...")

    best_loss = float('inf')
    best_x = x.detach().clone()

    for iteration in range(max_iters):
        optimizer.zero_grad()

        # Compute equilibrium errors
        ceq = balanced_trade_torch(x, data_np, param_np)

        # Loss: sum of squared residuals
        loss = (ceq ** 2).sum()

        # Backpropagate
        loss.backward()

        # Update
        optimizer.step()

        # Track best solution
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        # Print progress
        if iteration % 100 == 0:
            max_error = torch.abs(ceq).max().item()
            print(f"Iter {iteration:4d}: loss={loss.item():.2e}, max_error={max_error:.2e}")

        # Check convergence
        if max_error < 1e-4:
            print(f"\n✅ CONVERGED at iteration {iteration}!")
            return best_x.detach().numpy(), True

    print(f"\n⚠️  Reached max iterations. Best loss: {best_loss:.2e}")
    final_ceq = balanced_trade_torch(best_x, data_np, param_np)
    final_error = torch.abs(final_ceq).max().item()
    print(f"Final max error: {final_error:.2e}")

    return best_x.detach().numpy(), final_error < 1e-4


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch required. Install: pip install torch")
        return

    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(base_path, 'python_output')

    print("="*80)
    print("Multi-Sector Baseline: PyTorch Adam Optimizer")
    print("="*80)

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
    new_ustariff = new_ustariff_full[idx, :]

    id_US = 185 - 1  # Convert to 0-indexed (before filtering)
    id_US_new = np.where(idx == id_US + 1)[0][0]  # Find US index after filtering

    t_ji = np.zeros((N, N, K))
    t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
    t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])
    t_ji[id_US_new, id_US_new, :K-1] = 0

    # Load parameters from baseline model
    print("Loading baseline parameters...")
    phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
    phi = phi_data['phi'].values[idx]
    nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
    nu = nu_data['nu'].values[idx]

    # Calculate initial values
    E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
    Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
                nu * np.sum(np.sum(X_ji, axis=0), axis=1)
    T = E_i_multi - Y_i_multi

    # Calculate trade share and expenditure share parameters
    lambda_ji = X_ji / np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1))
    beta_i = np.tile(np.sum(X_ji, axis=0, keepdims=True), (N, 1, 1)) / \
             np.tile(E_i_multi.reshape(1, -1, 1), (N, 1, K))

    # Calculate sectoral income shares
    Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
    Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
    Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
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

    print("\n⚠️  This approach may take 5-10 minutes...")
    print("PyTorch's Adam optimizer uses momentum and adaptive learning rates")
    print("which can escape flat regions where traditional optimizers fail.\n")

    # Prepare data for PyTorch solver (as numpy arrays)
    data_np = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param_np = [eps_3D, kappa, psi, phi]

    print("\n" + "="*80)
    print("Attempting PyTorch Adam Optimization")
    print("="*80)

    # Try different learning rates
    learning_rates = [0.01, 0.001, 0.1]
    best_result = None
    best_error = float('inf')

    for lr in learning_rates:
        print(f"\n{'='*80}")
        print(f"Testing learning rate: {lr}")
        print(f"{'='*80}")

        x_solution, converged = solve_with_adam(data_np, param_np, max_iters=10000, lr=lr)

        # Check final error
        ceq_final = balanced_trade_torch(
            torch.tensor(x_solution, dtype=torch.float64),
            data_np,
            param_np
        )
        final_error = torch.abs(ceq_final).max().item()

        if final_error < best_error:
            best_error = final_error
            best_result = x_solution

        if converged:
            print(f"\n✅ SUCCESS with lr={lr}!")
            break

        print(f"\n⚠️ lr={lr} did not converge (error={final_error:.2e})")

    if best_error < 1e-4:
        print(f"\n{'='*80}")
        print(f"✅ CONVERGENCE ACHIEVED!")
        print(f"{'='*80}")
        print(f"Best error: {best_error:.2e}")
        print(f"\nSolution found! Saving results...")

        # Save results
        np.savez(os.path.join(output_dir, 'multisector_torch_solution.npz'),
                 x_solution=best_result,
                 error=best_error)
        print(f"Results saved to: {os.path.join(output_dir, 'multisector_torch_solution.npz')}")
    else:
        print(f"\n{'='*80}")
        print(f"❌ PyTorch Adam did not achieve convergence")
        print(f"{'='*80}")
        print(f"Best error achieved: {best_error:.2e}")
        print(f"Target error: 1e-4")
        print(f"\nThis suggests the problem may require:")
        print(f"  - Custom trust-region implementation")
        print(f"  - Better initial guess from MATLAB solution")
        print(f"  - Alternative problem formulation")

if __name__ == '__main__':
    if TORCH_AVAILABLE:
        main()
    else:
        print("\nPyTorch not available.")
        print("Install: pip3 install torch")
