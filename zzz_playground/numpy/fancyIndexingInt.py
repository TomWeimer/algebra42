import numpy as np

# A 3x3 array to index
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Two 2x2 integer arrays for coordinates
I1 = np.array([[0, 0],
               [2, 2]])
               
I2 = np.array([[0, 0],
               [1, 0]])

# This works and returns a 2x2 array
result = A[I1, I2]
print("Valid Indexing with Integers:")
print(result)