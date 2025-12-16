"""
Find the correct US index in the trade data
"""
import numpy as np
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), '..', '..')

# Load data
data_path = os.path.join(base_path, 'data', 'ITPDS', 'trade_ITPD.csv')
trade_data = pd.read_csv(data_path, header=None)

# Column 0 is exporter, column 1 is importer
# Get unique country codes (exporters)
countries = trade_data[0].unique()
print(f"Number of countries: {len(countries)}")
print(f"Countries (sorted): {sorted(countries)[:20]}...")

# Find USA
print(f"\nLooking for USA:")
for i, c in enumerate(countries):
    if 'US' in c.upper():
        print(f"  Index {i}: {c}")

# Find specific indices
usa_idx = np.where(countries == 'USA')[0]
print(f"\nUSA index: {usa_idx}")

# Also check if it's sorted differently
countries_sorted = sorted(countries)
usa_sorted_idx = countries_sorted.index('USA') if 'USA' in countries_sorted else None
print(f"USA index in sorted list: {usa_sorted_idx}")

# The countries in the data are in a specific order
# Let's see the first 20
print(f"\nFirst 20 countries (original order): {list(countries[:20])}")

# And around index 185
print(f"Countries around index 184-186:")
for i in range(182, 188):
    if i < len(countries):
        print(f"  Index {i}: {countries[i]}")

# Check what's at the actual US position
N = 194
K = 4
X = trade_data.iloc[:, 3].values
X_ji = X.reshape((N, N, K))

# Find US by looking for 'USA' in the data
usa_exporter_rows = trade_data[trade_data[0] == 'USA']
usa_importer_rows = trade_data[trade_data[1] == 'USA']
print(f"\nRows with USA as exporter: {len(usa_exporter_rows)}")
print(f"Rows with USA as importer: {len(usa_importer_rows)}")

# Get the index of USA in the unique countries list
usa_code_idx = list(countries).index('USA')
print(f"\nUSA index in original country order: {usa_code_idx}")

# Check domestic trade for USA at this index
print(f"\nUS domestic trade at index {usa_code_idx}:")
for k in range(K):
    print(f"  Sector {k}: X_ji[{usa_code_idx}, {usa_code_idx}, {k}] = {X_ji[usa_code_idx, usa_code_idx, k]:.2e}")

# Check some other major countries
print("\nLooking for China (CHN):")
if 'CHN' in countries:
    chn_idx = list(countries).index('CHN')
    print(f"  CHN index: {chn_idx}")
    print(f"  CHN domestic trade sector 0: {X_ji[chn_idx, chn_idx, 0]:.2e}")

print("\nLooking for Germany (DEU):")
if 'DEU' in countries:
    deu_idx = list(countries).index('DEU')
    print(f"  DEU index: {deu_idx}")
    print(f"  DEU domestic trade sector 0: {X_ji[deu_idx, deu_idx, 0]:.2e}")

# The answer: MATLAB uses alphabetically sorted countries?
# Let's check MATLAB code for how it determines country order
print("\n" + "=" * 60)
print("Checking if MATLAB uses different ordering")
print("=" * 60)

# Countries in alphabetical order
countries_alpha = sorted(countries)
usa_alpha_idx = countries_alpha.index('USA')
print(f"USA index in alphabetical order: {usa_alpha_idx}")

# Create mapping from alphabetical index to data index
data_to_alpha = {list(countries).index(c): countries_alpha.index(c) for c in countries}
alpha_to_data = {v: k for k, v in data_to_alpha.items()}

# Check if MATLAB index 185 (1-based) corresponds to alphabetical ordering
# MATLAB 185 = Python 184 (0-based in alphabetical order)
if 184 in alpha_to_data:
    matlab_185_country = countries_alpha[184]
    data_idx_for_185 = alpha_to_data[184]
    print(f"\nIf MATLAB uses alphabetical order:")
    print(f"  MATLAB index 185 (0-based: 184) = '{matlab_185_country}'")
    print(f"  Data index for this country: {data_idx_for_185}")

    # Check domestic trade
    print(f"  Domestic trade for '{matlab_185_country}':")
    for k in range(K):
        print(f"    Sector {k}: {X_ji[data_idx_for_185, data_idx_for_185, k]:.2e}")
