"""
Detailed test for the specific MATLAB line 37 conversion.

MATLAB line 37:
Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';

Breaking this down:
- (1-nu)' is a 1xN row vector (transpose of Nx1 nu)
- repmat((1-nu)',N,1) repeats it N times vertically to make NxN
- repmat((1-nu)',N,1).*X_ji is element-wise multiplication
- sum(...,2) sums across dimension 2 (rows) to get Nx1 vector
- sum(X_ji,1)' sums down columns (1xN), then transposes to Nx1
- nu.*sum(X_ji,1)' is element-wise multiplication of two Nx1 vectors
"""

import numpy as np

# Create test data
N = 3
X_ji = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
], dtype=float)

nu = np.array([0.1, 0.2, 0.3])

print("=" * 70)
print("TEST DATA:")
print("=" * 70)
print("X_ji (trade matrix, rows=exporters, cols=importers):")
print(X_ji)
print()
print("nu (Nx1 vector):")
print(nu)
print()

print("=" * 70)
print("MATLAB LINE 37 BREAKDOWN:")
print("=" * 70)
print("Y_i = sum( repmat((1-nu)',N,1).*X_ji,2) + nu.*sum(X_ji,1)';")
print()

# Part 1: repmat((1-nu)',N,1)
print("Step 1: (1-nu) =", 1 - nu)
print("  This is Nx1 vector in MATLAB")
print()

print("Step 2: (1-nu)' is transpose, making it 1xN row vector")
print("  In numpy: (1-nu).reshape(1, -1) =")
print("  ", (1 - nu).reshape(1, -1))
print()

print("Step 3: repmat((1-nu)',N,1) repeats row vector N times DOWN")
print("  Makes NxN matrix where each row is (1-nu)'")
nu_2D_matlab = np.tile((1 - nu).reshape(1, -1), (N, 1))
print(nu_2D_matlab)
print()

print("Step 4: repmat((1-nu)',N,1).*X_ji (element-wise multiplication)")
weighted_X = nu_2D_matlab * X_ji
print(weighted_X)
print()

print("Step 5: sum( repmat((1-nu)',N,1).*X_ji, 2)")
print("  sum(..., 2) sums along dimension 2 (ACROSS rows)")
print("  In numpy: sum(weighted_X, axis=1)")
term1 = np.sum(weighted_X, axis=1)
print("  Result:", term1)
print()

# Part 2: nu.*sum(X_ji,1)'
print("Step 6: sum(X_ji,1) sums DOWN columns (dimension 1)")
print("  Gives 1xN row vector of column totals")
col_sums = np.sum(X_ji, axis=0)
print("  In numpy: sum(X_ji, axis=0) =", col_sums)
print()

print("Step 7: sum(X_ji,1)' transposes to Nx1 column vector")
print("  In numpy, it's already 1D, so no change needed")
print("  Result:", col_sums)
print()

print("Step 8: nu.*sum(X_ji,1)' element-wise multiplication")
term2 = nu * col_sums
print("  nu * sum(X_ji, axis=0) =", term2)
print()

print("Step 9: Final Y_i = term1 + term2")
Y_i = term1 + term2
print("  Y_i =", Y_i)
print()

print("=" * 70)
print("CRITICAL FINDING:")
print("=" * 70)
print("MATLAB: sum( repmat((1-nu)',N,1).*X_ji,2)")
print("  -> np.sum(np.tile((1-nu).reshape(1, -1), (N, 1)) * X_ji, axis=1)")
print("  Result:", term1)
print()
print("MATLAB: nu.*sum(X_ji,1)'")
print("  -> nu * np.sum(X_ji, axis=0)")
print("  Result:", term2)
print()
print("Combined Y_i:", Y_i)
print()

print("=" * 70)
print("CHECKING PYTHON CODE LINES 188-189:")
print("=" * 70)
print("Python code has:")
print("  Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=0) +")
print("        nu * np.sum(X_ji, axis=0)")
print()
print("This uses axis=0 for the first sum, which is WRONG!")
print()
print("WRONG: axis=0 gives column sums:", np.sum(weighted_X, axis=0))
print("RIGHT: axis=1 gives row sums:   ", np.sum(weighted_X, axis=1))
print()

print("=" * 70)
print("THE BUG:")
print("=" * 70)
print("Line 188-189 in main_baseline.py should be:")
print("  Y_i = np.sum(np.tile((1 - nu).reshape(1, -1), (N, 1)) * X_ji, axis=1) +")
print("        nu * np.sum(X_ji, axis=0)")
print()
print("Currently it has axis=0 which gives the WRONG result!")
