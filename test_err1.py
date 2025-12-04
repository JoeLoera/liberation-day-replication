"""
Test for ERR1 calculation - checking axis mapping.

MATLAB line 329:
ERR1 = sum((1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;

Where nu_2D = repmat(nu',N,1) makes NxN matrix with rows = nu'
"""

import numpy as np

# Test data
N = 3
X_ji_new = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
], dtype=float)

nu = np.array([0.1, 0.2, 0.3])
w_i_h = np.ones(N)
L_i_h = np.ones(N)
Y_i = np.ones(N)

print("=" * 70)
print("MATLAB LINE 329 BREAKDOWN:")
print("=" * 70)
print("ERR1 = sum((1-nu_2D).*X_ji_new,2) + sum(nu_2D.*X_ji_new,1)' - w_i_h.*L_i_h.*Y_i;")
print()

# Create nu_2D
nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
print("nu_2D = repmat(nu',N,1):")
print(nu_2D)
print()

# First term: sum((1-nu_2D).*X_ji_new,2)
term1_weighted = (1 - nu_2D) * X_ji_new
print("(1-nu_2D).*X_ji_new:")
print(term1_weighted)
print()

print("sum((1-nu_2D).*X_ji_new, 2) sums along dimension 2 (ACROSS rows)")
print("  numpy equivalent: axis=1")
term1 = np.sum(term1_weighted, axis=1)
print("  Result:", term1)
print()

# Second term: sum(nu_2D.*X_ji_new,1)'
term2_weighted = nu_2D * X_ji_new
print("nu_2D.*X_ji_new:")
print(term2_weighted)
print()

print("sum(nu_2D.*X_ji_new, 1) sums along dimension 1 (DOWN columns)")
print("  numpy equivalent: axis=0")
term2 = np.sum(term2_weighted, axis=0)
print("  Result:", term2)
print()

print("sum(nu_2D.*X_ji_new, 1)' transposes to column vector")
print("  In numpy, already 1D array")
print()

# Final calculation
ERR1 = term1 + term2 - w_i_h * L_i_h * Y_i
print("ERR1 = term1 + term2 - w_i_h.*L_i_h.*Y_i")
print("     =", term1, "+", term2, "- 1")
print("     =", ERR1)
print()

print("=" * 70)
print("CHECKING PYTHON CODE LINES 88-89:")
print("=" * 70)
print("Python code has:")
print("  ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=0) +")
print("         np.sum(nu_2D * X_ji_new, axis=1) - w_i_h * L_i_h * Y_i")
print()

python_term1 = np.sum(term1_weighted, axis=0)
python_term2 = np.sum(term2_weighted, axis=1)
python_ERR1 = python_term1 + python_term2 - w_i_h * L_i_h * Y_i

print("Python term1 (axis=0):", python_term1)
print("Python term2 (axis=1):", python_term2)
print("Python ERR1:", python_ERR1)
print()

print("=" * 70)
print("THE BUG:")
print("=" * 70)
print("MATLAB: sum((1-nu_2D).*X_ji_new,2) -> axis=1 (row sums)")
print("Python has: axis=0 (column sums) -> WRONG!")
print()
print("MATLAB: sum(nu_2D.*X_ji_new,1)' -> axis=0 (column sums)")
print("Python has: axis=1 (row sums) -> WRONG!")
print()
print("The axes are SWAPPED in the Python code!")
print()
print("Correct Python code should be:")
print("  ERR1 = np.sum((1 - nu_2D) * X_ji_new, axis=1) +")
print("         np.sum(nu_2D * X_ji_new, axis=0) - w_i_h * L_i_h * Y_i")
