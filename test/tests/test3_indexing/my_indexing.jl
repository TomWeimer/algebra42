using Algebra42
using .TestData
using .TestUtils

using Debugger


function logIndexing(name, result, write_log_fn)
    write_log_fn("\n" * name)
    write_content(result, write_log_fn)
end

function test_shape_incompatibility(name, data::TestDataArray, write_log_fn, indices...)
    A = NDArray{Int}(data.matrix)

    passed = false
    try
        tmp = A[indices...]
        logIndexing(name, tmp, write_log_fn)
    catch e
        passed = e isa DomainError
    end
    
    @assert passed
end

function test_fancy_indexing(name, data::TestDataArray, write_log_fn, indices...)
    A = NDArray{Int}(data.matrix)
    got = A[indices...]
    logIndexing(name, got, write_log_fn)
end

function run_tests_my_indexing(write_log_fn)


# Two 2x2 integer arrays for coordinates
    # I1 = [1 1; 3 3]
    # I2 = [1 1; 2 1]

    # multi = Algebra42.new_multi_iter(I1, I2)

    # for (a, b) in multi
    #     write_log_fn(string(a) * "  " * string(b))
    # end
    ndarray1 = NDArray{Int}([1 1; 1 1])
    ndarray2 = NDArray{Int}([2 2; 2 2])

    ndarray3 = Algebra42.add(ndarray1, ndarray2)

    logIndexing("array: ", ndarray3, write_log_fn)
    
    #println("Fancy Indexing:")
   # test_fancy_indexing("Int indexing", array_4D_1, write_log_fn, [1, 2], [1 2; 2 1])
    #test_fancy_indexing("Bool mask 1", vector_1, write_log_fn, [true, false, true, true])
    #test_fancy_indexing("Bool mask 2", matrix_1, write_log_fn, [true false true; true true false])
    #test_fancy_indexing("Bool mask 3", matrix_1, write_log_fn,  [false false false; false false false])
    #test_fancy_indexing("Bool mask 4", array_3D_1, write_log_fn,[false, true], [false, true], [false, true])
    # TODO make possible nested arrray as indices
    #test_fancy_indexing("Int indexing", array_4D_1, write_log_fn, [0, 1], [[0, 1], [1, 0]])
    #test_fancy_indexing("Bool mask 5", array_3D_1, write_log_fn, [false, false], [false, false], [false, false])
    #test_fancy_indexing("Bool indexing", array_4D_1, write_log_fn, [[true, false], [false, true]], [[false, true], [true, false]])
    #test_shape_incompatibility("Shape invalid", array_4D_1, write_log_fn, [0, 1, 2], [[0, 1], [1, 0]])
    #test_fancy_indexing("Mixed indexing 1", array_4D_1, write_log_fn, [[true, false], [true, false]], [[0, 1], [0, 1]] )
    #test_shape_incompatibility("Mixed invalid", array_4D_1, write_log_fn, [[false, false], [false, false]], [[0, 1], [1, 0]])
end







# function testFancyIndexingEmptyBooleanMask(ndarray)
#   # Test an empty boolean mask
#     empty_mask = [false, false, false, false, false]
#     should =  ndarray[empty_mask] == []

#     # Test a full boolean mask
#     full_mask = [true, true, true, true, true]
#     content =  ndarray[full_mask] == ndarray

#     # Test with a different size mask (this will throw a BoundsError)
#     error = exceptionPassed(() -> ndarray[[true, false]], 1, DomainError)

#     return should && content && error
# end

# function testFancyIndexingMultiDim(my_2d_array)

#     row_mask = [true, false, true]
#     passed1 = compareByIndex(my_2d_array[row_mask, :], [1 2 3; 7 8 9])

#     col_mask = [true, false, true]
#     passed2 = compareByIndex(my_2d_array[:, col_mask], [1 3; 4 6; 7 9])

#     row_indices = [1, 3]
#     passed3 = compareByIndex(my_2d_array[row_indices, :],  [1 2 3; 7 8 9])

#     col_indices = [1, 3]
#     passed4 = compareByIndex(my_2d_array[:, col_indices], [1 3; 4 6; 7 9])

#     passed5 = compareByIndex(my_2d_array[row_indices, col_indices], [1 3; 7 9])

#     # passed3 = compareByIndex(my_2d_array[[1, 3], [1, 3]], [1 3; 7 9])

#     return passed1 && passed2 && passed3 && passed4 && passed5
# end

# function testFancyIndexing3d(my_3d_array)

#         println("data: ", my_3d_array)
#         # --- Indexing on Dimension 1 ---
#         dim1_mask = [true, false]
#         dim1_indices = [2]

#         passed1 = compareByIndex(my_3d_array[dim1_mask, :, :], el3d_array[[1], :,:])
#         passed1 || println("test1:\nexpected: ", el3d_array[[1], :,:], " got: ", my_3d_array[dim1_mask, :, :])

#         passed2 =  compareByIndex(my_3d_array[dim1_indices, :, :], [2 4 6 8; 10 12 14 16; 18 20 22 24])
#         passed2 || println("test2:\nexpected: ", [2 4 6 8; 10 12 14 16; 18 20 22 24], " got: ", my_3d_array[dim1_indices, :, :])


#         # --- Indexing on Dimension 2 ---
#         dim2_mask = [true, false, true]
#         dim2_indices = [1, 3]

#         passed3 = compareByIndex(my_3d_array[:, dim2_mask, :], [1 5 9 13; 2 6 10 14])
#         passed3 || println("test3:\nexpected: ", [1 5 9 13; 2 6 10 14], " got: ", my_3d_array[:, dim2_mask, :])

#         passed4 =  compareByIndex(my_3d_array[:, dim2_indices, :], [1 5 9 13; 2 6 10 14])
#         passed4 || println("test4\nexpected: ", [1 5 9 13; 2 6 10 14], " got: ", my_3d_array[:, dim2_indices, :])

#         # --- Indexing on Dimension 3 ---
#         dim3_mask = [false, true, false, true]
#         dim3_indices = [2, 4]

#         passed5::Bool = false

#         try
#             passed5 = compareByIndex(my_3d_array[:, :, dim3_mask], expected)
#             passed5 = false
#         catch e
#             passed5 = e isa DomainError
#         end

#         passed6::Bool = false

#         try
#             passed6 = compareByIndex(my_3d_array[:, :, dim3_indices], expected)
#             passed6 = false
#         catch e
#             passed6 = e isa DomainError
#         end

#         if !(passed1 && passed2 && passed3 && passed4 && passed5 && passed6)
#             println("passsed 1: ", passed1, " passed 2: ", passed2, " passed 3: ", passed3, " passed4: ", passed4, " passed 5: ", passed5, " passed6: ", passed6 )
#         end

#         return passed1 && passed2 && passed3 && passed4 && passed5 && passed6
#     end


#         #--------------------------------------------------------------------------------------------------------------------------

#    function mixedAndCombinedIndexing(my_3d_array)
#         # --- Boolean and Integer Indexing ---
#         row_mask = [true, false]
#         col_indices = [1, 3]
#         depth_mask = [true, false, true, false]

#         passed1 = compareByIndex(my_3d_array[row_mask, col_indices, depth_mask], [1 5 9 13; 2 6 10 14])

#         # --- Using arrays for all dimensions ---
#         row_indices = [1, 2]
#         col_indices = [3, 1]
#         depth_indices = [4, 2]

#         passed2 = compareByIndex(my_3d_array[row_indices, col_indices, depth_indices], [16 10; 24 18])

#         # --- Edge case with empty result ---
#         passed3 =  isempty(my_3d_array[:, [false, false, false], :])

#         return passed1 && passed2 && passed3
#     end

#     #--------------------------------------------------------------------------------------------------------------------------

#     function correctnessAndMutability(my_3d_array)
#         # Fancy indexing returns a copy, not a view
#         fancy_slice = my_3d_array[[1], :, :]
#         fancy_slice[1] = 999
#         return  my_3d_array[1] != 999
#     end
