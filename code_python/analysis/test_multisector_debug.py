"""
Quick debug script to test multi-sector equilibrium function
"""
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function
from sub_multisector_baseline import balanced_trade_multisector

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load data (simplified version of main script)
print("Loading data...")

# Load sectoral trade data
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path)

# Filter for year 2018 and aggregate by sector
trade_2018 = trade_data[trade_data['year'] == 2018].copy()

# Get country indices
countries = trade_2018['exporter'].unique()
N = len(countries)
K = 4  # Number of sectors

print(f"N = {N} countries, K = {K} sectors")

# Create trade matrix X_ji (simplified - just load saved values)
output_dir = os.path.join(base_path, 'python_output')

# Load saved baseline parameters
phi_data = pd.read_csv(os.path.join(output_dir, 'phi_values.csv'))
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))

phi = phi_data['phi'].values
nu = nu_data['nu'].values

# Load saved E_i and Y_i from baseline
baseline_results = np.load(os.path.join(output_dir, 'baseline_results.npz'))
E_i_baseline = baseline_results['E_i']
Y_i_baseline = baseline_results['Y_i']

print(f"E_i loaded: shape {E_i_baseline.shape}")
print(f"Y_i loaded: shape {Y_i_baseline.shape}")

# For multi-sector, we need to load the actual sectoral trade flows
# Let me create dummy data to test the function structure

# Create minimal test data
E_i_multi = E_i_baseline[:N]
Y_i_multi = Y_i_baseline[:N]

# Create dummy arrays for testing
lambda_ji = np.random.rand(N, N, K) * 0.1
lambda_ji = lambda_ji / np.sum(lambda_ji, axis=0, keepdims=True)  # Normalize

beta_i = np.random.rand(N, N, K) * 0.3
beta_i = beta_i / np.sum(beta_i, axis=(0, 2), keepdims=True)  # Normalize

ell_ik = np.random.rand(N, 1, K) * 0.3
ell_ik = ell_ik / np.sum(ell_ik, axis=2, keepdims=True)  # Normalize to sum to 1

t_ji = np.zeros((N, N, K))  # No tariffs for initial test

T = E_i_multi - Y_i_multi  # Trade imbalance

# Parameters
eps = np.array([3.3, 3.8, 4.1, 3.0]).reshape(1, 1, K)
kappa = 0.5
psi = 0.67/4
phi_param = phi[:N]

data = [N, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu[:N], T]
param = [eps, kappa, psi, phi_param]

# Test with initial guess
print("\nTesting equilibrium function with initial guess (all ones)...")
x0 = np.ones(3*N + N*K)

# Call the function
ceq, results, d_trade = balanced_trade_multisector(x0, data, param)

print(f"\nEquilibrium errors:")
print(f"  Max absolute error: {np.max(np.abs(ceq)):.2e}")
print(f"  Mean absolute error: {np.mean(np.abs(ceq)):.2e}")
print(f"  Median absolute error: {np.median(np.abs(ceq)):.2e}")
