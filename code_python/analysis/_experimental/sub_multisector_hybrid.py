"""
Hybrid PyTorch + scipy solver
Strategy: Use PyTorch Adam to escape the plateau, then scipy to refine
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

from scipy.optimize import fsolve, root

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


def balanced_trade_numpy(x, data, param):
    """
    Numpy version for scipy solvers
    """
    N, K = data[0], data[1]
    E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data[2:]
    eps, kappa, psi, phi = param

    # Extract and ensure positive
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:2*N])
    L_i_h = np.abs(x[2*N:3*N])
    ell_ik_h = np.abs(x[3*N:]).reshape(N, 1, K)

    # Construct 3D arrays
    w_i_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    L_ik_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(1, -1, 1), (N, 1, K))

    # Price index
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

    # Equilibrium equations
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
    return ceq


def solve_extended_pytorch(data_np, param_np, max_iters=20000, lr=0.001):
    """
    Extended PyTorch optimization with adaptive learning rate decay
    """
    N, K = data_np[0], data_np[1]

    print("\nOptimizing with PyTorch Adam...")
    print(f"Max iterations: {max_iters}")
    print(f"Initial learning rate: {lr}\n")

    # Initial guess
    x = torch.ones(3*N + N*K, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    # Learning rate scheduler - reduce LR when plateaued
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
    )

    best_loss = float('inf')
    best_x = x.detach().clone()
    plateau_counter = 0

    for iteration in range(max_iters):
        optimizer.zero_grad()
        ceq = balanced_trade_torch(x, data_np, param_np)
        loss = (ceq ** 2).sum()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()
            plateau_counter = 0
        else:
            plateau_counter += 1

        # Update learning rate based on loss
        scheduler.step(loss.item())

        if iteration % 500 == 0 or iteration < 100 and iteration % 20 == 0:
            max_error = torch.abs(ceq).max().item()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iter {iteration:5d}: loss={loss.item():.2e}, max_error={max_error:.2e}, lr={current_lr:.2e}", flush=True)

        # Check convergence
        if torch.abs(ceq).max().item() < 1e-4:
            print(f"\n✅ CONVERGED at iteration {iteration}!")
            return best_x.detach().numpy(), torch.abs(ceq).max().item()

        # Early stopping if completely plateaued
        if plateau_counter > 2000:
            print(f"\n⚠️  Plateaued for {plateau_counter} iterations, stopping early")
            break

    final_ceq = balanced_trade_torch(best_x, data_np, param_np)
    final_error = torch.abs(final_ceq).max().item()
    print(f"\nFinal error: {final_error:.2e}")

    return best_x.detach().numpy(), final_error


def solve_hybrid(data_np, param_np, pytorch_iters=2000, scipy_method='lm'):
    """
    Hybrid approach: PyTorch first, then scipy refinement
    """
    N, K = data_np[0], data_np[1]

    print("="*80, flush=True)
    print("PHASE 1: PyTorch Adam Optimizer (Escape Plateau)", flush=True)
    print("="*80, flush=True)

    # Initial guess
    x = torch.ones(3*N + N*K, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)  # Use best learning rate from previous run

    best_loss = float('inf')
    best_x = x.detach().clone()

    for iteration in range(pytorch_iters):
        optimizer.zero_grad()
        ceq = balanced_trade_torch(x, data_np, param_np)
        loss = (ceq ** 2).sum()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x = x.detach().clone()

        if iteration % 200 == 0:
            max_error = torch.abs(ceq).max().item()
            print(f"Iter {iteration:4d}: loss={loss.item():.2e}, max_error={max_error:.2e}", flush=True)

    # Get PyTorch solution
    x_pytorch = best_x.detach().numpy()
    ceq_pytorch = balanced_trade_torch(best_x, data_np, param_np)
    error_pytorch = torch.abs(ceq_pytorch).max().item()

    print(f"\nPyTorch Result: error = {error_pytorch:.2e}")

    print("\n" + "="*80)
    print("PHASE 2: scipy Refinement (Polish Solution)")
    print("="*80)

    # Use PyTorch solution as initial guess for scipy
    def system(x):
        return balanced_trade_numpy(x, data_np, param_np)

    print(f"Using scipy.optimize.root with method='{scipy_method}'...")
    print(f"Initial guess: PyTorch solution (error={error_pytorch:.2e})")

    result = root(
        system,
        x_pytorch,
        method=scipy_method,
        options={'xtol': 1e-10, 'ftol': 1e-10}
    )

    x_final = result.x
    ceq_final = system(x_final)
    error_final = np.max(np.abs(ceq_final))

    print(f"\nscip Result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Final error: {error_final:.2e}")

    converged = error_final < 1e-4

    return x_final, error_final, converged


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch required. Install: pip install torch")
        return

    base_path = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(base_path, 'python_output')

    print("="*80)
    print("Multi-Sector Hybrid Solver: PyTorch + scipy")
    print("="*80)

    # Load data (same as before)
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

    # Calculate trade shares
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
    eps = np.append(eps, 3.0)
    eps_3D = np.tile(eps.reshape(1, 1, -1), (N, N, 1))

    # Standard parameters
    kappa = 0.5
    psi = 0.67 / 4

    # Prepare data
    data_np = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
    param_np = [eps_3D, kappa, psi, phi]

    print("\nRunning hybrid solver...")
    print("This will use PyTorch to escape the plateau, then scipy to refine.\n")

    # Try pure PyTorch with many more iterations
    print("\n" + "="*80)
    print("Testing Extended PyTorch Optimization (20,000 iterations)")
    print("="*80)

    x_solution, final_error = solve_extended_pytorch(data_np, param_np, max_iters=20000)

    print("\n" + "="*80)
    converged = final_error < 1e-4
    if converged:
        print("✅ CONVERGENCE ACHIEVED!")
        print("="*80)
        print(f"Final error: {final_error:.2e}")

        # Save results
        np.savez(os.path.join(output_dir, 'multisector_hybrid_solution.npz'),
                 x_solution=x_solution,
                 error=final_error)
        print(f"\nResults saved to: multisector_hybrid_solution.npz")
    else:
        print("❌ Did not achieve full convergence")
        print("="*80)
        print(f"Final error: {final_error:.2e}")
        print(f"Target: 1e-4")
        print(f"\nImprovement from initial (972): {972/final_error:.1f}x better")


if __name__ == '__main__':
    if TORCH_AVAILABLE:
        main()
    else:
        print("\nPyTorch not available.")
        print("Install: pip3 install torch")
