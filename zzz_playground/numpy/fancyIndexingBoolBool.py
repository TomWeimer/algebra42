import numpy as np


# A 4D array (2x2x2x2)
A = np.arange(1, 17).reshape(2, 2, 2, 2)
print("4D Array A:")
print(A)

# Two 2x2 boolean arrays
B1 = np.array([[True, False],
               [True, False]])
B2 = np.array([[True, True],
               [False, False]])

# This causes a TypeError
print("\nInvalid Indexing with Booleans:")
try:
    result = A[B1, B2]
except TypeError as e:
    print(f"Error: {e}")