"""
Verify the reshape ordering bug for X_ji
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

print("=" * 60)
print("Testing reshape ordering")
print("=" * 60)

# C-order (Python default)
X_ji_c = X.reshape((N, N, K))  # Default C-order

# F-order (MATLAB-like)
X_ji_f = X.reshape((N, N, K), order='F')

# Get country indices
countries = trade_data[0].unique()
usa_idx = list(countries).index('USA')
abw_idx = list(countries).index('ABW')
print(f"USA index: {usa_idx}")
print(f"ABW index: {abw_idx}")

# Check US domestic trade
print(f"\n=== US DOMESTIC TRADE ===")
print(f"C-order reshape (default):")
for k in range(K):
    print(f"  Sector {k}: X_ji_c[{usa_idx}, {usa_idx}, {k}] = {X_ji_c[usa_idx, usa_idx, k]:.2e}")

print(f"\nF-order reshape (MATLAB-like):")
for k in range(K):
    print(f"  Sector {k}: X_ji_f[{usa_idx}, {usa_idx}, {k}] = {X_ji_f[usa_idx, usa_idx, k]:.2e}")

# Verify against the actual CSV values
print(f"\n=== VERIFICATION ===")
# Find the row for USA,USA,Agriculture in the CSV
usa_usa_agri = trade_data[(trade_data[0] == 'USA') & (trade_data[1] == 'USA') & (trade_data[2] == 'Agriculture')]
print(f"CSV value for USA,USA,Agriculture: {usa_usa_agri.iloc[0, 3]:.2e}")
print(f"X_ji_c[{usa_idx}, {usa_idx}, 0]: {X_ji_c[usa_idx, usa_idx, 0]:.2e}")
print(f"X_ji_f[{usa_idx}, {usa_idx}, 0]: {X_ji_f[usa_idx, usa_idx, 0]:.2e}")

# Also verify the file index
usa_usa_agri_idx = usa_usa_agri.index[0]
print(f"\nCSV row index for USA,USA,Agriculture: {usa_usa_agri_idx}")
print(f"X[{usa_usa_agri_idx}] = {X[usa_usa_agri_idx]:.2e}")

# Theoretical flat index for X_ji[usa_idx, usa_idx, 0]:
# C-order: flat = usa_idx*N*K + usa_idx*K + 0 = usa_idx*(N*K + K)
flat_c = usa_idx*N*K + usa_idx*K + 0
print(f"\nTheoretical flat index (C-order): {flat_c}")
print(f"X[{flat_c}] = {X[flat_c] if flat_c < len(X) else 'out of range'}")

# F-order: flat = usa_idx + usa_idx*N + 0*N*N = usa_idx*(1 + N)
flat_f = usa_idx + usa_idx*N + 0*N*N
print(f"Theoretical flat index (F-order): {flat_f}")
print(f"X[{flat_f}] = {X[flat_f] if flat_f < len(X) else 'out of range'}")

# The file ordering is: for each sector, for each importer, for each exporter
# So file index = exporter + importer*N + sector*N*N
# This matches F-order!

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("The data file is ordered as: exporter varies fastest, then importer, then sector")
print("This matches Fortran (column-major) order, NOT C (row-major) order!")
print("The correct reshape is: X_ji = X.reshape((N, N, K), order='F')")
