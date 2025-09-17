import numpy as np

# A 3x3 array to index
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# A 3x3 boolean mask
B_mask = np.array([[True, False, True],
                   [False, True, False],
                   [True, False, True]])

# This works and returns a 1D array of selected elements
result = A[B_mask]
print("\nValid Indexing with a Single Boolean Mask:")
print(result)