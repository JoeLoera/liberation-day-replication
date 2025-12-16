"""
Verify the MATLAB approach: set tariffs BEFORE filtering, then filter both X_ji and t_ji
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

# Load data with F-order reshape
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N_orig = 194
K = 4
X_ji_orig = X.reshape((N_orig, N_orig, K), order='F')

# Load tariffs
tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
new_ustariff = pd.read_csv(tariff_path).values

print("=" * 60)
print("MATLAB approach: Set tariffs on full matrix, then filter")
print("=" * 60)

# US index in original data (MATLAB: id_US = 185, Python: 184)
id_US = 184

# Set tariffs on full matrix FIRST (like MATLAB lines 15-19)
t_ji_orig = np.zeros((N_orig, N_orig, K))
for k in range(K-1):
    t_ji_orig[:, id_US, k] = new_ustariff.flatten()
t_ji_orig[:, id_US, :K-1] = np.maximum(0.1, t_ji_orig[:, id_US, :K-1])
t_ji_orig[id_US, id_US, :K-1] = 0

print(f"Tariffs set on column {id_US} (USA)")
print(f"Non-zero tariffs before filter: {np.count_nonzero(t_ji_orig)}")

# Find problematic countries (like MATLAB lines 21-23)
problematic_id = np.sum(np.all(X_ji_orig == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N_orig), ID)
N = len(idx)

print(f"Problematic countries: {ID}")
print(f"N after filtering: {N}")

# Find US in filtered index
id_US_new = np.where(idx == id_US)[0][0]
print(f"id_US_new: {id_US_new}")

# Filter BOTH X_ji and t_ji (like MATLAB lines 26-34)
X_ji = np.zeros((N, N, K))
t_ji = np.zeros((N, N, K))
for k in range(K):
    X_ji[:, :, k] = X_ji_orig[np.ix_(idx, idx, [k])].reshape(N, N)
    t_ji[:, :, k] = t_ji_orig[np.ix_(idx, idx, [k])].reshape(N, N)

print(f"\nAfter filtering:")
print(f"Non-zero tariffs: {np.count_nonzero(t_ji)}")
print(f"Tariffs on US column (sector 0): non-zero = {np.count_nonzero(t_ji[:, id_US_new, 0])}")
print(f"US tariff on sector 0: [{t_ji[:, id_US_new, 0].min():.4f}, {t_ji[:, id_US_new, 0].max():.4f}]")

# Verify US domestic trade
countries = trade_data[0].unique()
print(f"\nUS domestic trade:")
for k in range(K):
    print(f"  Sector {k}: {X_ji[id_US_new, id_US_new, k]:.2e}")

# Now compare with the CURRENT Python approach
print("\n" + "=" * 60)
print("Current Python approach: Filter first, then set tariffs")
print("=" * 60)

# Reset and do it the current Python way
X_ji_orig2 = X.reshape((N_orig, N_orig, K), order='F')
problematic_id = np.sum(np.all(X_ji_orig2 == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
idx = np.setdiff1d(np.arange(N_orig), ID)
N = len(idx)

X_ji_py = np.zeros((N, N, K))
for k in range(K):
    X_ji_py[:, :, k] = X_ji_orig2[np.ix_(idx, idx, [k])].reshape(N, N)

# Filter tariffs to match filtered countries
new_ustariff_filtered = new_ustariff[idx, :]
id_US_new_py = np.where(idx == 184)[0][0]

# Set tariffs on filtered matrix
t_ji_py = np.zeros((N, N, K))
t_ji_py[:, id_US_new_py, :K-1] = np.tile(new_ustariff_filtered, (1, K-1))
t_ji_py[:, id_US_new_py, :K-1] = np.maximum(0.1, t_ji_py[:, id_US_new_py, :K-1])
t_ji_py[id_US_new_py, id_US_new_py, :K-1] = 0

print(f"Non-zero tariffs: {np.count_nonzero(t_ji_py)}")
print(f"Tariffs on US column (sector 0): non-zero = {np.count_nonzero(t_ji_py[:, id_US_new_py, 0])}")
print(f"US tariff on sector 0: [{t_ji_py[:, id_US_new_py, 0].min():.4f}, {t_ji_py[:, id_US_new_py, 0].max():.4f}]")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
diff_t = np.abs(t_ji - t_ji_py)
print(f"Max difference in tariffs: {diff_t.max():.10f}")
print(f"Are tariffs identical? {np.allclose(t_ji, t_ji_py)}")

diff_x = np.abs(X_ji - X_ji_py)
print(f"Max difference in X_ji: {diff_x.max():.10f}")
print(f"Are X_ji identical? {np.allclose(X_ji, X_ji_py)}")
