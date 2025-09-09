
import numpy as np

a = np.arange(12).reshape(3,4)

# Fancy indexing with 1D arrays
rows = np.array([0, 2])
cols = np.array([1, 3])
a[rows, cols]  # returns array([1, 11]), 1D in this case

# Fancy indexing with 2D arrays
rows2 = np.array([[0, 1], [1, 2]])
cols2 = np.array([[0, 1], [2, 3]])

print(rows2)
print("\n")
print(cols2)

a[rows2, cols2]  # returns a 2x2 array

print(a[rows2, cols2])