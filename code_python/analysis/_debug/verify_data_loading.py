"""
Verify data loading matches what MATLAB would see
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load data exactly as the main code does
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)
X = trade_data.iloc[:, 3].values
N = 194
K = 4
X_ji = X.reshape((N, N, K), order='F')

# Get countries
countries = trade_data[0].unique()
usa_idx = list(countries).index('USA')
chn_idx = list(countries).index('CHN')

print("=" * 60)
print("Data verification after X_ji fix")
print("=" * 60)

print(f"\nUS domestic trade (X_ji[USA, USA, :]):")
for k in range(K):
    sector_names = ['Agriculture', 'Manufacturing', 'Mining', 'Services']
    print(f"  {sector_names[k]}: {X_ji[usa_idx, usa_idx, k]:.2e}")

# Verify against CSV
print(f"\nVerification against CSV:")
for k, sector in enumerate(['Agriculture', 'Manufacturing', 'Mining and Energy', 'Services']):
    csv_val = trade_data[(trade_data[0] == 'USA') & (trade_data[1] == 'USA') & (trade_data[2] == sector)].iloc[0, 3]
    arr_val = X_ji[usa_idx, usa_idx, k]
    match = "✓" if np.isclose(csv_val, arr_val) else "✗"
    print(f"  {sector}: CSV={csv_val:.2e}, Array={arr_val:.2e} {match}")

# Check US total trade
print(f"\nUS total imports (column sum):")
us_imports = np.sum(X_ji[:, usa_idx, :], axis=0)
for k in range(K):
    print(f"  Sector {k}: {us_imports[k]:.2e}")

print(f"\nUS total exports (row sum):")
us_exports = np.sum(X_ji[usa_idx, :, :], axis=0)
for k in range(K):
    print(f"  Sector {k}: {us_exports[k]:.2e}")

# Check lambda_ji (trade shares)
lambda_ji = X_ji / np.sum(X_ji, axis=0, keepdims=True)
print(f"\nUS domestic share (lambda_ji[USA, USA, :]):")
for k in range(K):
    print(f"  Sector {k}: {lambda_ji[usa_idx, usa_idx, k]:.4f}")

# China share of US market
print(f"\nChina share of US market:")
for k in range(K):
    print(f"  Sector {k}: {lambda_ji[chn_idx, usa_idx, k]:.4f}")

# Total trade volumes
print(f"\nTotal global trade by sector:")
for k in range(K):
    total = np.sum(X_ji[:, :, k])
    domestic = np.sum(np.diag(X_ji[:, :, k]))
    intl = total - domestic
    print(f"  Sector {k}: Total={total:.2e}, Domestic={domestic:.2e}, International={intl:.2e}")
