

import numpy as np
# A 4D array (2x2x2x2)
A = np.arange(1, 17).reshape(2, 2, 2, 2)
print("4D Array A:")
print(A)


# Two 2x2 boolean arrays
B1 = np.array([ [True, False], [True, False] ]) # <- result1 slices result2 row

B2 = np.array([                                 # <- result1 row    result2 slices     
                [1, 1], 
                [0, 0]
            ])

try:
    print("Valid Indexing with Booleans:")
    result = A[B1, B2]
    print(f"\nA[B1, B2] = \n{result}")

    slice1 = A[0, 0, :, :]  # row 1  == B2(0,0)  # <- from A[1, 0] row 0 ([0, 0] == ( [ B1[0, 0] == true, B2[0, 0] ) 
    print(f"slice1 = \n{slice1}")
    slice2 = A[1, 0, :, :]  # row 1  == B2(0, 1)    # <- from A[0, 1] row 0 ([0, 1] == ( [B1[0, 1] == false, B2[0, 1] ) 
    print(f"\nslice2 = \n{slice2}")
    slice3 = A[0, 0, :, :] # row 0  == B2(1, 0) # <- from A[1, 1] row 0 ([1, 1] == ( [B1[1, 0] == true, B2[1, 0] ) 
    print(f"\nslice3 = \n{slice3}")
    slice4 = A[1,0, :, :] # row 0    ==  B2(1, 1)     # <- from A[0, 1] row 1 ([0, 1] == ( [B1[1, 1] = false, B2[1, 1] ) 
    print(f"\nslice4 = \n{slice4}")


    result2 = A[B2, B1]
    print(f"\n-----------------------------------------------")
    print(f"\nA[B1, B2] = \n{result2}")

    slice1 = A[1, 0, :, :]  # row 0 select [B2(0, 0), B1(0, 0)] rows are by default taken at index 0
    print(f"slice1 = \n{slice1}")
    slice2 = A[1, 1, :, :]  # row 0  [B2(0, 1), B1(0, 1)] rows are by default taken at index 0
    print(f"\nslice2 = \n{slice2}")
    slice3 = A[0, 0, :, :] # row  0   [B2(1, 0), B1(1, 0)] rows ...
    print(f"\nslice3 = \n{slice3}")
    slice4 = A[0, 1, :, :] # row  0    [B2(1, 1), B1(1, 1)] rows ...
    print(f"\nslice4 = \n{slice4}")
    
    

except TypeError as e:
    print("Invalid Indexing with Booleans:")
    print(f"\nError: {e}")

