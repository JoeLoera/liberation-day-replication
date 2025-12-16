"""
Debug US data - check why US domestic share is zero
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load data
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
print(f"Trade data shape: {trade_data.shape}")
print(f"Trade data columns: {trade_data.columns.tolist()}")
print(f"First few rows:")
print(trade_data.head(10))

X = trade_data.iloc[:, 3].values
N = 194
K = 4
print(f"\nX shape: {X.shape}, expected: {N*N*K}")
X_ji = X.reshape((N, N, K))

# Check US index
# MATLAB uses 1-based indexing, US is country 185
id_US_matlab = 185  # 1-indexed
id_US_python = 185 - 1  # 0-indexed = 184

print(f"\nUS index (MATLAB): {id_US_matlab}")
print(f"US index (Python): {id_US_python}")

# Check US domestic trade
print(f"\n=== BEFORE FILTERING ===")
print(f"US domestic trade (diagonal):")
for k in range(K):
    print(f"  Sector {k}: X_ji[{id_US_python}, {id_US_python}, {k}] = {X_ji[id_US_python, id_US_python, k]:.2e}")

print(f"\nUS total imports (column):")
for k in range(K):
    print(f"  Sector {k}: sum(X_ji[:, {id_US_python}, {k}]) = {X_ji[:, id_US_python, k].sum():.2e}")

print(f"\nUS total exports (row):")
for k in range(K):
    print(f"  Sector {k}: sum(X_ji[{id_US_python}, :, {k}]) = {X_ji[id_US_python, :, k].sum():.2e}")

# Check if any countries have zero trade
problematic_id = np.sum(np.all(X_ji == 0, axis=0), axis=1)
ID = np.where(problematic_id == 1)[0]
print(f"\nProblematic countries (indices): {ID}")
print(f"Number of problematic countries: {len(ID)}")
print(f"Is US in problematic list? {id_US_python in ID}")

# Filter
idx = np.setdiff1d(np.arange(N), ID)
N_new = len(idx)
print(f"\nFiltered N: {N_new}")

# Find US in filtered index
print(f"\nChecking US in idx:")
print(f"  idx contains {id_US_python}? {id_US_python in idx}")
if id_US_python in idx:
    id_US_new = np.where(idx == id_US_python)[0][0]
    print(f"  id_US_new = {id_US_new}")
else:
    print(f"  US was filtered out!")

# Check what the MATLAB code expects
print(f"\n=== CHECKING MATLAB's US INDEX ===")
print(f"MATLAB line: id_US_new = find(idx == 185)")
print(f"In Python: idx == 185 means looking for value 185 in idx")
print(f"  idx == 185: {idx == 185}")
print(f"  np.where(idx == 185): {np.where(idx == 185)}")

# The issue: MATLAB uses 1-indexed, so idx contains 1-indexed values!
# When MATLAB does idx = setdiff(1:N, ID), idx contains values 1, 2, ..., N (excluding IDs)
# When MATLAB does find(idx == 185), it's looking for the value 185 in idx

# But in Python, we use 0-indexed arrays
# np.arange(N) gives [0, 1, ..., 193]
# So idx contains 0-indexed values

print(f"\n=== THE BUG ===")
print(f"Python idx values: {idx[:10]}... (0-indexed)")
print(f"MATLAB idx values would be: 1, 2, 3, ... (1-indexed)")
print(f"MATLAB looks for idx == 185 (1-indexed US)")
print(f"Python should look for idx == 184 (0-indexed US)")
print(f"  idx == 184: where = {np.where(idx == 184)}")

# Correct US index
id_US_python_correct = 184
if id_US_python_correct in idx:
    id_US_new_correct = np.where(idx == id_US_python_correct)[0][0]
    print(f"\nCORRECT id_US_new = {id_US_new_correct}")

    # Filter data
    X_new = np.zeros((N_new, N_new, K))
    for k in range(K):
        X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N_new, N_new)

    print(f"\n=== AFTER FILTERING ===")
    print(f"US domestic trade (diagonal):")
    for k in range(K):
        print(f"  Sector {k}: X_new[{id_US_new_correct}, {id_US_new_correct}, {k}] = {X_new[id_US_new_correct, id_US_new_correct, k]:.2e}")

    # Trade shares
    lambda_ji = X_new / np.tile(np.sum(X_new, axis=0, keepdims=True), (N_new, 1, 1))
    print(f"\nUS domestic share (lambda_ji diagonal):")
    for k in range(K):
        print(f"  Sector {k}: lambda_ji[{id_US_new_correct}, {id_US_new_correct}, {k}] = {lambda_ji[id_US_new_correct, id_US_new_correct, k]:.4f}")

print("\n" + "=" * 60)
print("CHECKING THE ACTUAL CODE's BUG")
print("=" * 60)

# Check what the current code does:
# id_US = 185 - 1  # = 184
# id_US_new = np.where(idx == id_US + 1)[0][0]  # looking for 185 in idx!

id_US = 185 - 1  # 184
print(f"Current code: id_US = 185 - 1 = {id_US}")
print(f"Current code: looking for idx == id_US + 1 = {id_US + 1}")
print(f"  np.where(idx == {id_US + 1}): {np.where(idx == id_US + 1)}")

# The issue is clear: Python idx contains 0-indexed values (0, 1, ..., 193)
# But the code is looking for value 185, which doesn't exist if indexing is 0-based!

# Let me check if idx actually contains 1-indexed or 0-indexed values
print(f"\nActual idx values:")
print(f"  min(idx) = {idx.min()}")
print(f"  max(idx) = {idx.max()}")
print(f"  Does idx contain 0? {0 in idx}")
print(f"  Does idx contain 194? {194 in idx}")
