"""
Debug the tariff setup and compare with MATLAB
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(base_path, 'python_output')

# Load data
print("Loading data...")
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
N_new = len(idx)

X_new = np.zeros((N_new, N_new, K))
for k in range(K):
    X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N_new, N_new)
X_ji = X_new
N = N_new

print(f"N = {N} countries")

# Load tariffs
tariff_path = os.path.join(base_path, 'data', 'base_data', 'tariffs.csv')
tariff_data_full = pd.read_csv(tariff_path)
new_ustariff_full = tariff_data_full.values
new_ustariff = new_ustariff_full[idx, :]

print(f"\nnew_ustariff shape: {new_ustariff.shape}")
print(f"new_ustariff range: [{new_ustariff.min():.4f}, {new_ustariff.max():.4f}]")
print(f"new_ustariff non-zero count: {np.count_nonzero(new_ustariff)}")

id_US = 185 - 1
id_US_new = np.where(idx == id_US + 1)[0][0]
print(f"\nUS index (original): {id_US}")
print(f"US index (after filtering): {id_US_new}")

# Check what MATLAB does with repmat
# MATLAB: t_ji(:,id_US,1:K-1)=repmat(new_ustariff, [1 1 K-1]);
# This takes new_ustariff (N x 1) and tiles to (N x 1 x K-1)
# Then assigns to t_ji(:, id_US, 1:K-1) which is (N, 1, K-1) sliced from (N, N, K)

print("\n" + "=" * 60)
print("Checking tariff assignment")
print("=" * 60)

t_ji = np.zeros((N, N, K))

# MATLAB: t_ji(:,id_US,1:K-1)=repmat(new_ustariff, [1 1 K-1])
# new_ustariff is (N, 1), repmat to (N, 1, K-1)
# Then assign to the slice

# In Python, we need to tile new_ustariff correctly
print(f"new_ustariff shape before tile: {new_ustariff.shape}")

# The MATLAB repmat([N x 1], [1 1 K-1]) creates [N x 1 x (K-1)]
# In Python, np.tile should do the same
tiled_tariff = np.tile(new_ustariff.reshape(N, 1, 1), (1, 1, K-1))
print(f"tiled_tariff shape: {tiled_tariff.shape}")

# Now assign to t_ji[:, id_US_new, :K-1]
# This is selecting column id_US_new and first K-1 sectors
# Shape should be (N, K-1)

print(f"t_ji[:, id_US_new, :K-1] shape: {t_ji[:, id_US_new, :K-1].shape}")

# The tiled_tariff is (N, 1, K-1), we need to squeeze the middle dimension
t_ji[:, id_US_new, :K-1] = tiled_tariff.squeeze(axis=1)
print(f"After assignment, t_ji[:, id_US_new, :K-1] shape: {t_ji[:, id_US_new, :K-1].shape}")

# Apply minimum tariff
t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])

# Zero out US-to-US
t_ji[id_US_new, id_US_new, :K-1] = 0

print(f"\nTariff stats:")
print(f"  t_ji[:, id_US_new, 0] (sector 0): [{t_ji[:, id_US_new, 0].min():.4f}, {t_ji[:, id_US_new, 0].max():.4f}], mean={t_ji[:, id_US_new, 0].mean():.4f}")
print(f"  t_ji[:, id_US_new, 1] (sector 1): [{t_ji[:, id_US_new, 1].min():.4f}, {t_ji[:, id_US_new, 1].max():.4f}], mean={t_ji[:, id_US_new, 1].mean():.4f}")
print(f"  t_ji[:, id_US_new, 2] (sector 2): [{t_ji[:, id_US_new, 2].min():.4f}, {t_ji[:, id_US_new, 2].max():.4f}], mean={t_ji[:, id_US_new, 2].mean():.4f}")
print(f"  t_ji[:, id_US_new, 3] (sector 3, services): [{t_ji[:, id_US_new, 3].min():.4f}, {t_ji[:, id_US_new, 3].max():.4f}]")

print(f"\n  US tariff on imports from China (country 31 before filter):")
id_CHN = np.where(idx == 34)[0]  # China is country 34 in original
if len(id_CHN) > 0:
    id_CHN = id_CHN[0]
    print(f"  China index: {id_CHN}")
    print(f"  t_ji[id_CHN, id_US_new, :] = {t_ji[id_CHN, id_US_new, :]}")

# Check total number of non-zero tariffs
print(f"\nTotal non-zero tariffs: {np.count_nonzero(t_ji)}")
print(f"Total tariff revenue potential: {np.sum(t_ji):.4f}")

# Now let's trace through what the code was doing wrong
print("\n" + "=" * 60)
print("What the BUGGY code was doing:")
print("=" * 60)

t_ji_buggy = np.zeros((N, N, K))
# Buggy code: np.tile(new_ustariff, (1, 1, K-1))
# new_ustariff is (N, 1), np.tile(..., (1, 1, K-1)) would try to tile in 3D
# But np.tile adds dimensions at the beginning if needed
tiled_buggy = np.tile(new_ustariff, (1, 1, K-1))
print(f"Buggy tiled shape: {tiled_buggy.shape}")

# The issue: np.tile(new_ustariff, (1, 1, K-1)) where new_ustariff is (N, 1)
# This creates (1, N, K-1), not (N, 1, K-1)!
print(f"  Expected shape: (N, 1, K-1) = ({N}, 1, 3)")
print(f"  Actual buggy shape: {tiled_buggy.shape}")

# When assigned to t_ji[:, id_US_new, :K-1] which is (N, K-1):
# (1, N, K-1) can't broadcast to (N, K-1) properly...
# Actually let's see what happens
try:
    t_ji_buggy[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
    print("Assignment succeeded (shouldn't happen)")
except Exception as e:
    print(f"Assignment failed with error: {e}")

# Let me check what the original code actually does
print("\n" + "=" * 60)
print("Original code behavior:")
print("=" * 60)

# Original code from sub_multisector_baseline.py:
# t_ji[:, id_US_new, :K-1] = np.tile(new_ustariff, (1, 1, K-1))
# t_ji[:, id_US_new, :K-1] = np.maximum(0.1, t_ji[:, id_US_new, :K-1])

t_ji_orig = np.zeros((N, N, K))
print(f"new_ustariff shape: {new_ustariff.shape}")
print(f"t_ji[:, id_US_new, :K-1] shape: {t_ji_orig[:, id_US_new, :K-1].shape}")
print(f"np.tile(new_ustariff, (1, 1, K-1)) shape: {np.tile(new_ustariff, (1, 1, K-1)).shape}")

# The shapes don't match! This would be an error or unexpected broadcast
