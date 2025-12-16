"""
Debug the ell_ik calculation
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
N = len(idx)

X_new = np.zeros((N, N, K))
for k in range(K):
    X_new[:, :, k] = X_ji[np.ix_(idx, idx, [k])].reshape(N, N)
X_ji = X_new

# Load nu
nu_data = pd.read_csv(os.path.join(output_dir, 'nu_values.csv'))
nu = nu_data['nu'].values[idx]

# Calculate Y_i_multi (MATLAB line 48)
# Y_i = sum( repmat((1-nu)',N,1).*sum(X_ji,3) , 2) + nu.*sum(sum(X_ji,1),3)'
Y_i_multi = np.sum((1 - nu).reshape(-1, 1) * np.sum(X_ji, axis=2), axis=1) + \
            nu * np.sum(np.sum(X_ji, axis=0), axis=1)

print(f"\nY_i_multi shape: {Y_i_multi.shape}")
print(f"Y_i_multi range: [{Y_i_multi.min():.2e}, {Y_i_multi.max():.2e}]")

# Calculate Y_ik (MATLAB lines 55-58)
# Y_ik_p = sum( repmat((1-nu)',[ N 1 K]).* X_ji , 2)
# Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1)
# Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3])
# ell_ik = Y_ik./repmat( Y_i_multi, [1 1 K])

Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
print(f"\nY_ik_p shape: {Y_ik_p.shape}")  # Should be (N, 1, K)

Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
print(f"Y_ik_f shape: {Y_ik_f.shape}")  # Should be (1, N, K)

Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
print(f"Y_ik shape: {Y_ik.shape}")  # Should be (N, 1, K)

ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1, 1), (1, 1, K))
print(f"ell_ik shape: {ell_ik.shape}")  # Should be (N, 1, K)

# Check: sum of ell_ik over sectors should equal 1
sum_ell_ik = np.sum(ell_ik, axis=2)
print(f"\nsum(ell_ik, axis=2) shape: {sum_ell_ik.shape}")  # Should be (N, 1)
print(f"sum(ell_ik, axis=2) range: [{sum_ell_ik.min():.4f}, {sum_ell_ik.max():.4f}]")
print(f"sum(ell_ik, axis=2) should be 1.0 for all countries")

# Check: sum of Y_ik over sectors should equal Y_i_multi
sum_Y_ik = np.sum(Y_ik, axis=2).reshape(N)
print(f"\nsum(Y_ik, axis=2) vs Y_i_multi:")
print(f"  max difference: {np.max(np.abs(sum_Y_ik - Y_i_multi)):.2e}")

# Alternative verification: compute ell_ik directly from income shares
print("\n" + "=" * 60)
print("Alternative calculation of ell_ik:")
print("=" * 60)

# For each country i and sector k:
# Y_ik = sum over j of (1-nu_j) * X_ji_k + nu_i * E_ik
# where E_ik = sum over j of X_ji_k (imports from j to i in sector k)

# Actually, let me re-read the MATLAB code more carefully
# Y_ik_p = sum( repmat((1-nu)',[ N 1 K]).* X_ji , 2)
#   - repmat((1-nu)',[ N 1 K]) tiles (1-nu)' (which is 1xN) to create N x N x K
#   - This multiplies each row of X_ji by (1-nu)
#   - sum(..., 2) sums over the 2nd dimension (columns/importers)
#   - Result: (N x 1 x K) - sales from each exporter to all importers, weighted by (1-nu) of importer

# Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1)
#   - sum(X_ji, 1) sums over the 1st dimension (rows/exporters) -> (1 x N x K)
#   - repmat(nu',[1 1 K]) tiles nu' (1 x N) to create (1 x N x K)
#   - Multiplies total imports by nu
#   - Result: (1 x N x K) - total imports for each importer, weighted by nu

# Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3])
#   - permute([2 1 3]) swaps first two dimensions of Y_ik_f
#   - Y_ik_f becomes (N x 1 x K)
#   - Add to Y_ik_p (N x 1 x K) -> Result: (N x 1 x K)

# Wait, there's something wrong here. Let me trace through:
# - Y_ik_p is (N, 1, K) - for each exporter i and sector k, total sales weighted by (1-nu) of importers
# - Y_ik_f after permute is (N, 1, K) - but this is confusing because nu is for importers

# Let me re-examine MATLAB's indexing
# In MATLAB, repmat((1-nu)', [N 1 K]) where (1-nu)' is (1 x N):
#   - Tiles to (N x N x K)
#   - Each row of the N x N slice is (1-nu_1, 1-nu_2, ..., 1-nu_N)

# X_ji is (N x N x K) where X_ji(j, i, k) = trade from j to i in sector k
# MATLAB uses column-major, so X_ji(:,:,k) has exporters (j) in rows and importers (i) in columns

# repmat((1-nu)', [N 1 K]) .* X_ji
#   - For each k: (N x N) .* (N x N) element-wise
#   - (1-nu)'(j) = (1-nu_j) for the j-th column... wait, (1-nu)' is a row vector

# Actually in MATLAB:
# nu is (N x 1), nu' is (1 x N)
# repmat(nu', [N 1 K]) creates (N x N x K) where each row is nu'
# So element (j, i, k) is nu_i

# Therefore:
# Y_ik_p = sum( repmat((1-nu)',[ N 1 K]).* X_ji , 2)
#   - repmat((1-nu)', [N 1 K]): element (j, i, k) = (1 - nu_i)
#   - Element-wise multiply with X_ji: (1 - nu_i) * X_ji(j, i, k)
#   - Sum over dimension 2 (i): sum over all importers
#   - Result Y_ik_p(j, 1, k) = sum_i (1 - nu_i) * X_ji(j, i, k)
#   - This is the wage income of exporter j in sector k from selling to all importers

# Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1)
#   - sum(X_ji, 1): sum over exporters (j), result is (1 x N x K)
#   - sum(X_ji, 1)(1, i, k) = sum_j X_ji(j, i, k) = total imports by i in sector k
#   - repmat(nu', [1 1 K]): (1 x N x K) where element (1, i, k) = nu_i
#   - Element-wise multiply: nu_i * (total imports by i in sector k)
#   - Result Y_ik_f(1, i, k) = nu_i * sum_j X_ji(j, i, k)

# Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3])
#   - permute([2 1 3]) swaps dimensions 1 and 2
#   - Y_ik_f becomes (N x 1 x K) where Y_ik_f(i, 1, k) = nu_i * (total imports by i in sector k)
#   - Y_ik_p is (N x 1 x K) where Y_ik_p(j, 1, k) = sum_i (1 - nu_i) * X_ji(j, i, k)
#   - Adding them: Y_ik(i, 1, k) = [sum_j (1-nu_j) * X_ij(j, i, k)] + [nu_i * sum_j X_ji(j, i, k)]

# Wait, this is confusing. Let me use consistent notation.
# In the model, country i's income from sector k is:
#   Y_ik = (1 - nu_i) * (sales from i to all j in sector k) + nu_i * (total expenditure of i on sector k)

# Let me verify the Python implementation

# In Python:
# Y_ik_p = np.sum(np.tile((1 - nu).reshape(1, -1, 1), (N, 1, K)) * X_ji, axis=1, keepdims=True)
#   - (1 - nu).reshape(1, -1, 1) is (1, N, 1)
#   - np.tile(..., (N, 1, K)) creates (N, N, K)
#   - Element (j, i, k) = (1 - nu_i)
#   - Multiply by X_ji: (1 - nu_i) * X_ji(j, i, k)
#   - Sum over axis=1 (importers): sum_i (1 - nu_i) * X_ji(j, i, k)
#   - Result shape: (N, 1, K) where element (j, 0, k) = sum_i (1 - nu_i) * X_ji(j, i, k)

# Y_ik_f = np.tile(nu.reshape(1, -1, 1), (1, 1, K)) * np.sum(X_ji, axis=0, keepdims=True)
#   - np.sum(X_ji, axis=0, keepdims=True) sums over exporters, shape (1, N, K)
#   - Element (0, i, k) = sum_j X_ji(j, i, k) = total imports by i in sector k
#   - nu.reshape(1, -1, 1) is (1, N, 1)
#   - np.tile(..., (1, 1, K)) creates (1, N, K)
#   - Element (0, i, k) = nu_i
#   - Multiply: nu_i * (total imports by i in sector k)
#   - Result shape: (1, N, K)

# Y_ik = Y_ik_p + np.transpose(Y_ik_f, (1, 0, 2))
#   - Transpose Y_ik_f from (1, N, K) to (N, 1, K)
#   - Y_ik_f transposed: element (i, 0, k) = nu_i * (total imports by i in sector k)
#   - Add to Y_ik_p: (N, 1, K) + (N, 1, K)
#   - But wait! Y_ik_p(j, 0, k) is about EXPORTER j, Y_ik_f(i, 0, k) is about IMPORTER i
#   - After adding, element (i, 0, k) = [sum_j (1-nu_j) * X_ji(i, j, k)] + [nu_i * sum_j X_ji(j, i, k)]
#   - This doesn't make sense! The first term sums over j for sales FROM i,
#   - but the second term sums over j for imports TO i

# I think the issue is that my understanding of X_ji indexing is wrong.
# Let me check: in the paper and MATLAB code, X_ji typically means trade FROM j TO i
# But in the code, how is it indexed?

# Looking at the MATLAB code again:
# X_ji = reshape(X, N, N, K)
# trade = X_ji.*(1-eye(N));  % This zeros out diagonal (domestic trade)
# sum(sum(X_ji,1),3)' = total imports (sum over exporters j and sectors k)
# sum(sum(X_ji,2),3) = total exports (sum over importers i and sectors k)

# So X_ji(j, i, k) = flow FROM j TO i in sector k

# Let me verify the income accounting:
# Country i's income Y_i = sum_k Y_ik
# Y_ik = (1 - nu_i) * (sales from i to all j) + nu_i * (expenditure of i on k)
#      = (1 - nu_i) * sum_j X_ij(i, j, k) + nu_i * sum_j X_ji(j, i, k)

# In MATLAB indexing:
# Sales from i to all j = sum over j: X_ji(i, j, k) -- but wait, j is the exporter in X_ji!
# Actually, X_ij (from i to j) would be X_ji(i, j, k) if we swap the roles
# No wait, that's the same thing written differently.

# I think the confusion is:
# X_ji(j, i, k) = X^k_{ji} = flow from j to i in sector k
# So X_ji(i, j, k) = X^k_{ij} = flow from i to j in sector k (i sells to j)

# Therefore:
# Income of i in sector k =
#   (1 - nu_j) * X^k_{ij} summed over j (value added from sales, net of intermediate input share)
#   + nu_i * E_ik (resource income from expenditure on sector k)

# Wait, that's still not right. Let me look at the model more carefully.

# From the paper, income consists of:
# 1. Labor income: w_i * L_i
# 2. Resource income: proportional to expenditure E_i with share nu_i

# Sectoral income Y_ik:
# Y_ik = value added created by country i in sector k
#      = (1 - nu_i) * (total sales of country i's sector k production)
#        where the (1 - nu_i) factor accounts for intermediate inputs

# Let me just verify numerically
print("\nNumerical verification:")
print(f"Sum of Y_ik over k for country 0: {np.sum(Y_ik[0, :, :]):.4e}")
print(f"Y_i_multi for country 0: {Y_i_multi[0]:.4e}")
print(f"Ratio: {np.sum(Y_ik[0, :, :]) / Y_i_multi[0]:.4f}")

# Check for all countries
ratios = np.sum(Y_ik, axis=2).reshape(N) / Y_i_multi
print(f"\nRatio of sum(Y_ik) / Y_i_multi for all countries:")
print(f"  Min: {ratios.min():.4f}")
print(f"  Max: {ratios.max():.4f}")
print(f"  Mean: {ratios.mean():.4f}")

# If ratio is not 1.0, then sum(ell_ik) won't be 1.0

# Calculate ell_ik correctly: ell_ik should be Y_ik / Y_i so that sum equals 1
ell_ik_corrected = Y_ik / np.sum(Y_ik, axis=2, keepdims=True)
sum_ell_corrected = np.sum(ell_ik_corrected, axis=2)
print(f"\nCorrected ell_ik (normalized within sector):")
print(f"  sum(ell_ik_corrected, axis=2): [{sum_ell_corrected.min():.4f}, {sum_ell_corrected.max():.4f}]")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print(f"The issue is that sum(Y_ik) != Y_i_multi")
print(f"This means the MATLAB formula for Y_i_multi and Y_ik are inconsistent,")
print(f"OR we need to use sum(Y_ik) as the denominator instead of Y_i_multi.")
