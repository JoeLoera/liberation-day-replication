"""
Compare MATLAB vs Python tariff setup exactly
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load data
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K))

tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
tariff_data = pd.read_csv(tariff_path)
new_ustariff = tariff_data.values

print("=" * 60)
print("MATLAB approach: Set tariffs BEFORE filtering")
print("=" * 60)

id_US = 185 - 1  # 0-indexed
t_ji_matlab = np.zeros((N, N, K))
# MATLAB: t_ji(:,id_US,1:K-1)=repmat(new_ustariff, [1 1 K-1])
# new_ustariff is (194, 1), repmat to (194, 1, 3)
for k in range(K-1):
    t_ji_matlab[:, id_US, k] = new_ustariff.flatten()
t_ji_matlab[:, id_US, :K-1] = np.maximum(0.1, t_ji_matlab[:, id_US, :K-1])
t_ji_matlab[id_US, id_US, :K-1] = 0

# Now filter
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N), ID)
N_new = len(idx)

# Filter both X_ji and t_ji
X_ji_new = np.zeros((N_new, N_new, K))
t_ji_new_matlab = np.zeros((N_new, N_new, K))
for k in range(K):
    X_ji_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N_new, N_new)
    t_ji_new_matlab[:, :, k] = t_ji_matlab[np.ix_(idx, idx, [k])].reshape(N_new, N_new)

id_US_new = np.where(idx == id_US)[0][0]

print(f"Original N = 194, Filtered N = {N_new}")
print(f"id_US (original) = {id_US}, id_US_new = {id_US_new}")
print(f"\nTariffs on imports to US (MATLAB approach):")
print(f"  t_ji[:, id_US_new, 0]: non-zero = {np.count_nonzero(t_ji_new_matlab[:, id_US_new, 0])}")
print(f"  t_ji[:, id_US_new, 0] range: [{t_ji_new_matlab[:, id_US_new, 0].min():.4f}, {t_ji_new_matlab[:, id_US_new, 0].max():.4f}]")
print(f"  t_ji[:, id_US_new, 0] mean (excl US): {t_ji_new_matlab[np.arange(N_new) != id_US_new, id_US_new, 0].mean():.4f}")

print("\n" + "=" * 60)
print("Python approach: Filter FIRST, then set tariffs")
print("=" * 60)

# Reset
X_ji = X.reshape((194, 194, K))
new_ustariff_full = tariff_data.values

# Filter first
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(194), ID)
N_new = len(idx)

X_ji_filtered = np.zeros((N_new, N_new, K))
for k in range(K):
    X_ji_filtered[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N_new, N_new)

# Filter tariffs to match
new_ustariff_filtered = new_ustariff_full[idx, :]

id_US_orig = 185 - 1
id_US_new = np.where(idx == id_US_orig)[0][0]

# Now set tariffs on filtered data
t_ji_python = np.zeros((N_new, N_new, K))
t_ji_python[:, id_US_new, :K-1] = np.tile(new_ustariff_filtered, (1, K-1))
t_ji_python[:, id_US_new, :K-1] = np.maximum(0.1, t_ji_python[:, id_US_new, :K-1])
t_ji_python[id_US_new, id_US_new, :K-1] = 0

print(f"\nTariffs on imports to US (Python approach):")
print(f"  t_ji[:, id_US_new, 0]: non-zero = {np.count_nonzero(t_ji_python[:, id_US_new, 0])}")
print(f"  t_ji[:, id_US_new, 0] range: [{t_ji_python[:, id_US_new, 0].min():.4f}, {t_ji_python[:, id_US_new, 0].max():.4f}]")
print(f"  t_ji[:, id_US_new, 0] mean (excl US): {t_ji_python[np.arange(N_new) != id_US_new, id_US_new, 0].mean():.4f}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

diff = np.abs(t_ji_new_matlab - t_ji_python)
print(f"Max difference between MATLAB and Python tariffs: {diff.max():.10f}")
print(f"Are they identical? {np.allclose(t_ji_new_matlab, t_ji_python)}")

if not np.allclose(t_ji_new_matlab, t_ji_python):
    print("\nDifferences found:")
    nonzero_diff = np.where(diff > 1e-10)
    for i in range(min(10, len(nonzero_diff[0]))):
        j, k, l = nonzero_diff[0][i], nonzero_diff[1][i], nonzero_diff[2][i]
        print(f"  t_ji[{j}, {k}, {l}]: MATLAB={t_ji_new_matlab[j,k,l]:.6f}, Python={t_ji_python[j,k,l]:.6f}")
