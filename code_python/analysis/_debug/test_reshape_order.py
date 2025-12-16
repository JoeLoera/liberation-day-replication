"""
Test reshape order: MATLAB uses Fortran (column-major), Python uses C (row-major)
"""
import numpy as np

N = 4  # countries
K = 3  # sectors

# Create a (N, 1, K) array with distinct values
a = np.arange(N * K).reshape((N, 1, K))
print("Original array a with shape (N, 1, K) = (4, 1, 3):")
for i in range(N):
    print(f"  Country {i}: {a[i, 0, :]}")

# C-order (Python default) reshape to (N*K,)
a_c = a.reshape(N * K, order='C')
print("\nC-order reshape (Python default):")
print(f"  {a_c}")
print("  Elements 0-2: country 0, all sectors")
print("  Elements 3-5: country 1, all sectors")
print("  ...")

# Fortran-order reshape to (N*K,)
a_f = a.reshape(N * K, order='F')
print("\nFortran-order reshape (MATLAB default):")
print(f"  {a_f}")
print("  Elements 0-3: all countries, sector 0")
print("  Elements 4-7: all countries, sector 1")
print("  ...")

# In MATLAB code: ERR1(N,1) = ... replaces the N-th element
# In Fortran order, element N-1 (0-indexed) corresponds to:
#   - Country N-1, sector 0
# Let's verify:
print("\nElement indices in Fortran order:")
for i in range(N * K):
    country = i % N
    sector = i // N
    print(f"  a_f[{i}] = a[{country}, 0, {sector}] = {a_f[i]}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("In MATLAB, ERR1(N) is country N-1 (0-indexed), sector 0")
print("In Python with C-order, ERR1[N-1] is country (N-1)//K, sector (N-1)%K")
print(f"\nFor N={N}, K={K}:")
print(f"  MATLAB ERR1({N}) = country {N-1}, sector 0 (element {N-1} in F-order)")
print(f"  Python ERR1[{N-1}] = country {(N-1)//K}, sector {(N-1)%K}")
print(f"\nThese are DIFFERENT unless we use F-order reshape in Python!")

# Fix: use F-order reshape
print("\n" + "=" * 60)
print("FIX: Use F-order reshape in Python to match MATLAB")
print("=" * 60)
print("Change: ERR1 = (Y_ik_cf - Y_ik*Y_ik_h).reshape(N*K)")
print("To:     ERR1 = (Y_ik_cf - Y_ik*Y_ik_h).reshape(N*K, order='F')")
