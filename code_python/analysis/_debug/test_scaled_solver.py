"""
Test solver with equation scaling
Normalize all equations to similar magnitudes to improve convergence
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sub_multisector_baseline import balanced_trade_multisector

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

print("="*80)
print("Testing Solver with Equation Scaling")
print("="*80)

# Load data (same as before)
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

data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu, T]
param = [eps_3D, kappa, psi, phi]

x0 = np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(N*K)])

# Compute scaling factors from initial evaluation
print("\nComputing equation scaling factors...")
ceq0, _, _ = balanced_trade_multisector(x0, data, param)

# Compute RMS (root mean square) for each equation block
ERR1_0 = ceq0[:N*K]
ERR2_0 = ceq0[N*K:N*K+N]
ERR3_0 = ceq0[N*K+N:N*K+2*N]
ERR4_0 = ceq0[N*K+2*N:]

scale1 = np.sqrt(np.mean(ERR1_0**2)) if np.sqrt(np.mean(ERR1_0**2)) > 1e-10 else 1.0
scale2 = np.sqrt(np.mean(ERR2_0**2)) if np.sqrt(np.mean(ERR2_0**2)) > 1e-10 else 1.0
scale3 = np.sqrt(np.mean(ERR3_0**2)) if np.sqrt(np.mean(ERR3_0**2)) > 1e-10 else 1.0
scale4 = np.sqrt(np.mean(ERR4_0**2)) if np.sqrt(np.mean(ERR4_0**2)) > 1e-10 else 1.0

print(f"\nOriginal RMS magnitudes:")
print(f"  ERR1 (sectoral income):  {scale1:.2e}")
print(f"  ERR2 (total income):     {scale2:.2e}  ← Dominant!")
print(f"  ERR3 (labor supply):     {scale3:.2e}")
print(f"  ERR4 (sectoral shares):  {scale4:.2e}")

print(f"\nScale ratio (max/min): {max(scale1, scale2, scale3, scale4) / min(scale1, scale2, scale3, scale4):.2e}")

# Create scaled system
def scaled_syst(x):
    ceq, _, _ = balanced_trade_multisector(x, data, param)
    # Scale each equation block by its RMS
    ceq[:N*K] /= scale1
    ceq[N*K:N*K+N] /= scale2
    ceq[N*K+N:N*K+2*N] /= scale3
    ceq[N*K+2*N:] /= scale4
    return ceq

# Test scaled system
ceq_scaled = scaled_syst(x0)
ERR1_s = ceq_scaled[:N*K]
ERR2_s = ceq_scaled[N*K:N*K+N]
ERR3_s = ceq_scaled[N*K+N:N*K+2*N]
ERR4_s = ceq_scaled[N*K+2*N:]

print(f"\nScaled RMS magnitudes (should be ~1.0):")
print(f"  ERR1: {np.sqrt(np.mean(ERR1_s**2)):.2e}")
print(f"  ERR2: {np.sqrt(np.mean(ERR2_s**2)):.2e}")
print(f"  ERR3: {np.sqrt(np.mean(ERR3_s**2)):.2e}")
print(f"  ERR4: {np.sqrt(np.mean(ERR4_s**2)):.2e}")

# Solve scaled system
print("\n" + "="*80)
print("Solving with fsolve (hybr method) + equation scaling...")
print("="*80)

start_time = time.time()

x_sol, infodict, ier, mesg = fsolve(
    scaled_syst,
    x0,
    xtol=1e-10,
    maxfev=1000000,
    full_output=True
)

elapsed_time = time.time() - start_time

print(f"\nSolver completed in {elapsed_time:.1f} seconds")
print(f"Status (ier={ier}): {mesg}")
print(f"Function evaluations: {infodict['nfev']}")

# Check final error (unscaled)
ceq_final, results, d_trade = balanced_trade_multisector(x_sol, data, param)

print(f"\n" + "="*80)
print("RESULTS (unscaled equilibrium errors)")
print("="*80)

print(f"Max equilibrium error: {np.max(np.abs(ceq_final)):.2e}")
print(f"Mean equilibrium error: {np.mean(np.abs(ceq_final)):.2e}")

ERR1_f = ceq_final[:N*K]
ERR2_f = ceq_final[N*K:N*K+N]
ERR3_f = ceq_final[N*K+N:N*K+2*N]
ERR4_f = ceq_final[N*K+2*N:]

print(f"\nError breakdown:")
print(f"  ERR1 (sectoral income): max={np.max(np.abs(ERR1_f)):.2e}")
print(f"  ERR2 (total income):     max={np.max(np.abs(ERR2_f)):.2e}")
print(f"  ERR3 (labor supply):     max={np.max(np.abs(ERR3_f)):.2e}")
print(f"  ERR4 (sectoral shares):  max={np.max(np.abs(ERR4_f)):.2e}")

if np.max(np.abs(ceq_final)) < 1e-6:
    print("\n" + "="*80)
    print("✅ SUCCESS! Solver converged to target tolerance")
    print("="*80)
    print(f"\nUSA welfare change: {results[id_US_new, 0]:.2f}%")
    print(f"Global trade-to-GDP change: {d_trade:.2f}%")
    print(f"\nTarget values (from MATLAB Table 4, row 3):")
    print(f"  USA welfare: 0.60%")
    print(f"  Global trade: -5.5% (Table 11)")
else:
    print("\n" + "="*80)
    print("⚠️ Solver did not fully converge")
    print("="*80)
    print(f"Final error ({np.max(np.abs(ceq_final)):.2e}) exceeds target (1e-6)")

    if np.max(np.abs(ceq_final)) < 100:
        print("\n✓ Partial success: Error reduced from 972 to {:.2e}".format(np.max(np.abs(ceq_final))))
        print("  This is significant progress! Further refinement may work.")
