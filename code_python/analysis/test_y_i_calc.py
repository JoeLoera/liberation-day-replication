"""
Quick test to verify Y_i_multi calculation matches MATLAB
"""
import numpy as np
import pandas as pd
import os

# Set up base path
base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load sectoral trade data
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

print(f"N = {N}, K = {K}")
print(f"X_ji shape: {X_ji.shape}")

# Load nu parameter
output_dir = os.path.join(base_path, 'python_output')
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

print(f"nu shape: {nu.shape}")
print(f"nu min/max: {np.min(nu):.4f} / {np.max(nu):.4f}")

# Calculate E_i_multi (OLD way - should be same)
E_i_multi = np.sum(np.sum(X_ji, axis=0), axis=1)
print(f"\nE_i_multi: min={np.min(E_i_multi):.2e}, max={np.max(E_i_multi):.2e}, sum={np.sum(E_i_multi):.2e}")

# Calculate Y_i_multi (NEW corrected way)
# MATLAB: Y_i = sum( repmat((1-nu)',N,1).*sum(X_ji,3) , 2) + nu.*sum(sum(X_ji,1),3)'
Y_i_multi_new = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
                nu * np.sum(np.sum(X_ji, axis=0), axis=1)

print(f"Y_i_multi (NEW): min={np.min(Y_i_multi_new):.2e}, max={np.max(Y_i_multi_new):.2e}, sum={np.sum(Y_i_multi_new):.2e}")

# Check if T = E_i - Y_i makes sense
T = E_i_multi - Y_i_multi_new
print(f"\nT = E_i - Y_i:")
print(f"  min={np.min(T):.2e}, max={np.max(T):.2e}")
print(f"  sum(T) = {np.sum(T):.2e} (should be close to 0)")
print(f"  countries with T > 0: {np.sum(T > 0)} (importers)")
print(f"  countries with T < 0: {np.sum(T < 0)} (exporters)")

# Check that E_i and Y_i are in reasonable ranges
print(f"\nSanity checks:")
print(f"  E_i / Y_i ratio: min={np.min(E_i_multi / Y_i_multi_new):.3f}, max={np.max(E_i_multi / Y_i_multi_new):.3f}")
print(f"  |T| / E_i ratio: min={np.min(np.abs(T) / E_i_multi):.3f}, max={np.max(np.abs(T) / E_i_multi):.3f}")
