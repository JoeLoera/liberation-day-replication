"""
Test to verify MATLAB sum() to numpy sum() axis mapping.

This test creates a simple matrix and compares what MATLAB's sum(X,1) and sum(X,2)
do versus numpy's sum(X, axis=0) and sum(X, axis=1).
"""

import numpy as np

# Create a simple test matrix
# In trade context: rows = exporters, columns = importers
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Test Matrix X (3x3):")
print(X)
print()

# What MATLAB does:
# sum(X, 1) - sums along dimension 1 (DOWN columns), gives 1xN row vector of column totals
# sum(X, 2) - sums along dimension 2 (ACROSS rows), gives Nx1 column vector of row totals

print("=" * 70)
print("MATLAB BEHAVIOR:")
print("=" * 70)
print("sum(X, 1) - sums DOWN columns (collapses rows)")
print("  For each column j, sum over all rows i: sum_i X[i,j]")
print("  Result: [sum of col 0, sum of col 1, sum of col 2]")
print("  Expected: [12, 15, 18]  (1+4+7, 2+5+8, 3+6+9)")
print()
print("sum(X, 2) - sums ACROSS rows (collapses columns)")
print("  For each row i, sum over all columns j: sum_j X[i,j]")
print("  Result: [sum of row 0, sum of row 1, sum of row 2]")
print("  Expected: [6, 15, 24]  (1+2+3, 4+5+6, 7+8+9)")
print()

print("=" * 70)
print("NUMPY BEHAVIOR:")
print("=" * 70)
print("np.sum(X, axis=0) - sums DOWN axis 0 (collapses rows)")
print("  For each column j, sum over all rows i: sum_i X[i,j]")
print("  Result:", np.sum(X, axis=0))
print()
print("np.sum(X, axis=1) - sums ACROSS axis 1 (collapses columns)")
print("  For each row i, sum over all columns j: sum_j X[i,j]")
print("  Result:", np.sum(X, axis=1))
print()

print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("MATLAB sum(X, 1) = numpy sum(X, axis=0)  [both give COLUMN totals / imports]")
print("MATLAB sum(X, 2) = numpy sum(X, axis=1)  [both give ROW totals / exports]")
print()

# Now let's verify in the trade context
print("=" * 70)
print("TRADE MATRIX INTERPRETATION:")
print("=" * 70)
print("If X_ji is trade matrix where rows=exporters, columns=importers:")
print()
print("MATLAB: sum(X_ji, 1)' = column sums = total IMPORTS by each country")
print("  This transposes to make it a column vector")
print("  numpy equivalent: np.sum(X_ji, axis=0)")
print()
print("MATLAB: sum(X_ji, 2) = row sums = total EXPORTS by each country")
print("  Already a column vector")
print("  numpy equivalent: np.sum(X_ji, axis=1)")
print()

# Verify the specific case from the code
print("=" * 70)
print("VERIFICATION FOR MATLAB LINE 36:")
print("=" * 70)
print("MATLAB: E_i = sum(X_ji,1)' ")
print("  sum(X_ji,1) gives row vector of column totals (total imports)")
print("  The ' transposes it to column vector")
print()
print("Python: Should be E_i = np.sum(X_ji, axis=0)")
print("  This gives column totals (total imports) as 1D array")
print()

print("=" * 70)
print("VERIFICATION FOR MATLAB LINES 356-357:")
print("=" * 70)
print("MATLAB: sum(X_ji_new.*(1-eye(N)),2) - sums ACROSS rows (exports excluding diagonal)")
print("  numpy equivalent: np.sum(X_ji_new * (1-np.eye(N)), axis=1)")
print()
print("MATLAB: sum(X_ji_new.*(1-eye(N)),1) - sums DOWN columns (imports excluding diagonal)")
print("  numpy equivalent: np.sum(X_ji_new * (1-np.eye(N)), axis=0)")
print()

# Create test case for the actual problem
print("=" * 70)
print("REAL-WORLD SANITY CHECK:")
print("=" * 70)
print("Consider this trade matrix:")
print("     Country 0  Country 1  Country 2  (importers across columns)")
print("C0      10         20         30       <- Country 0 exports")
print("C1      40         50         60       <- Country 1 exports")
print("C2      70         80         90       <- Country 2 exports")
print("(exporters down rows)")
print()
print("Exports by each country (row sums, sum over columns):")
exports_by_country = np.sum(X * 10, axis=1)  # multiply by 10 to match the visual
print("  Country 0 exports:", 10+20+30, "= axis=1 sum of row 0")
print("  Country 1 exports:", 40+50+60, "= axis=1 sum of row 1")
print("  Country 2 exports:", 70+80+90, "= axis=1 sum of row 2")
print("  numpy: np.sum(X, axis=1) =", np.sum(X, axis=1))
print()
print("Imports by each country (column sums, sum over rows):")
imports_by_country = np.sum(X * 10, axis=0)
print("  Country 0 imports:", 10+40+70, "= axis=0 sum of col 0")
print("  Country 1 imports:", 20+50+80, "= axis=0 sum of col 1")
print("  Country 2 imports:", 30+60+90, "= axis=0 sum of col 2")
print("  numpy: np.sum(X, axis=0) =", np.sum(X, axis=0))
