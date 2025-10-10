using Algebra42
using .TestData
using .TestUtils

function logViewContent(name, data::Bool, write_log_fn)
    write_log_fn("\n" * name)
    write_log_fn(data ? "true" : "false")
end

function logViewContent(name, data, write_log_fn)
    write_log_fn("\n" * name)
    write_content(data, write_log_fn)
end


function test_basic_view2(write_log_fn)
    # -----------------------------
    # Basic contiguous view
    # -----------------------------
    A = Algebra42.reshape(0:15, (4, 4))
    v = Algebra42.create_view_from_indices(A, 2:3, 2:4)

    logViewContent("basic view 1", v, write_log_fn)

    v[1, 1] = -99

    logViewContent("basic view 1 (modified)", A[2, 2] == -99, write_log_fn)
end


function test_colon_view2(write_log_fn)
    # -----------------------------
    # Colon view
    # -----------------------------
    B = Algebra42.reshape(0:8, (3, 3))
    v_colon = Algebra42.create_view_from_indices(B, :, :)

    logViewContent("colon view 1", v_colon, write_log_fn)

    logViewContent("colon size ok: ", size(v_colon) == size(B), write_log_fn)

    logViewContent("same memory ok: ", pointer(v_colon) == pointer(B), write_log_fn)
end

function test_dimension_droping2(write_log_fn)
    # -----------------------------
    # Dimension dropping
    # -----------------------------
    A = Algebra42.reshape(0:15, (4, 4))

    row = Algebra42.create_view_from_indices(A, 2, :)
    col = Algebra42.create_view_from_indices(A, :, 3)

    logViewContent("row view:", row, write_log_fn)
    logViewContent("col view:", col, write_log_fn)

    logViewContent("row view size ok:", size(row) == (4,), write_log_fn)
    logViewContent("col view size ok:", size(col) == (4,), write_log_fn)

    logViewContent("row view access ok:", row[1] == A[2, 1], write_log_fn)
    logViewContent("col view access ok:", col[4] == A[4, 3], write_log_fn)
end

function test_strided_slice2(write_log_fn)
    
    A = Algebra42.reshape(0:15, (4, 4))

    v_stride = Algebra42.create_view_from_indices(A, 1:2:4, 1:2:4)

    println("v_stride: ", v_stride)

    logViewContent("strided slice view:", v_stride, write_log_fn)

    logViewContent("strided slice size ok:", size(v_stride) == (2, 2), write_log_fn)

    logViewContent("strided slice access ok 1:", v_stride[1, 1] == A[1, 1], write_log_fn)

    logViewContent("strided slice access ok 2:", v_stride[2, 2] == A[3, 3], write_log_fn)
    
    logViewContent("strided share memory:", pointer(A) == pointer(v_stride), write_log_fn)
end

function test_view_of_view2(write_log_fn)
    A = Algebra42.reshape(0:15, (4, 4))

    v1 = Algebra42.create_view_from_indices(A, 2:4, 2:4)

    logViewContent("view of view v1:", v1, write_log_fn)
    
    v2 = Algebra42.create_view_from_indices(v1, 2:3, 1:2)
    
    logViewContent("view of view v2:", v2, write_log_fn)

    logViewContent("view v2 size ok:", size(v2) == (2, 2), write_log_fn)

    logViewContent("view v2 access ok:",  v2[1, 1] == A[3, 2], write_log_fn)

    logViewContent("v2 share memory:",  pointer(A) == pointer(v2), write_log_fn)
end

function test_fancy_indexing_copy2(write_log_fn)
    A = Algebra42.reshape(0:15, (4, 4))
    rows = [1, 3, 4]
    cols = [2, 4]
    fancy = A[rows, :][:, cols]  # creates a copy

    logViewContent("fancy index: ", fancy, write_log_fn)

    logViewContent("fancy index is copy: ", !pointer(A) == pointer(fancy), fancy, write_log_fn)

    logViewContent("fancy index size ok: ", size(fancy) == (3, 2), fancy, write_log_fn)

    logViewContent("fancy index access 1 ok: ", fancy[1, 1] == A[1, 2], fancy, write_log_fn)

    logViewContent("fancy index access 2 ok: ", fancy[3, 2] == A[4, 4], fancy, write_log_fn)
end

function test_scalar_view2(write_log_fn)
    A = Algebra42.reshape(0:15, (4, 4))

    # -----------------------------
    # Scalar view
    # -----------------------------
    s = A[2, 3]

    logViewContent("scalar view: ", s,  write_log_fn)

    logViewContent("scalar content: ", s == 6,  write_log_fn)

    logViewContent("is numeric: ", typeof(s) <: Number,  write_log_fn)
end

function test_transpose_view2(write_log_fn)
   B = Algebra42.reshape(0:8, (3, 3))
    
    At = transpose(B)

    logViewContent("transpose view: ", At, write_log_fn)
    logViewContent("transpose is view: ", pointer(A) == pointer(At), write_log_fn)
    logViewContent("transpose size ok: ", size(At) == (3, 3), write_log_fn)
    logViewContent("transpose access ok: ", At[1, 2] == B[2, 1], write_log_fn)
end

function test_write_through_view2(write_log_fn)
    B = Algebra42.reshape(0:8, (3, 3))

    v_write = Algebra42.create_view_from_indices(B, 2:3, 2:3)

    logViewContent("write through view: ", v_write, write_log_fn)
    v_write[1, 1] = -42

    logViewContent("write through parent: ", B[2, 2] == -42, write_log_fn)
end

function test_trivial_view2(write_log_fn)
    B = Algebra42.reshape(0:8, (3, 3))

    v_trivial = Algebra42.create_view_from_indices(B, :, :)

    logViewContent("trivial view: ", v_trivial, write_log_fn)

    logViewContent("trivial view is parent: ", pointer(v_trivial) == pointer(B), write_log_fn)
end


function test_shape_and_strides_consistency2(write_log_fn)
    # -----------------------------
    # Shape and stride consistency
    # -----------------------------
    C = Algebra42.reshape(0:11, (3, 4))

    v_stride2 = Algebra42.create_view_from_indices(C, 1:2:size(C,1), :)
    
    logViewContent("v_stride2 view: ", v_stride2, write_log_fn)

    logViewContent("v_stride2 size: ", size(v_stride2) == (2, 4), write_log_fn)

    logViewContent("v_stride2 strides ok: ", stride(v_stride2, 1) == 2 * stride(C, 1), write_log_fn)
end

function test_fancy_indexing_diagonal2(write_log_fn)
    C = Algebra42.reshape(0:11, (3, 4))

    v_fancy_diag = C[[1,2], [3,4]]

    logViewContent("v_fancy_diag view: ", v_fancy_diag, write_log_fn)
    logViewContent("v_fancy_diag is copy: ", !pointer(C) == pointer(v_fancy_diag), write_log_fn)
    logViewContent("v_fancy_diag size: ", size(v_fancy_diag) == (2,), write_log_fn)
end

function test_nested_view2(write_log_fn)

    A = Algebra42.reshape(0:15, (4, 4))
    # -----------------------------
    # Nested view does not copy
    # -----------------------------
    v_nested1 = Algebra42.create_view_from_indices(A, 2:size(A, 1), :)

    logViewContent("v_nested1 view: ", v_nested1, write_log_fn)


    v_nested2 = Algebra42.create_view_from_indices(v_nested1, :, 3:(v_nested1, 2))

    logViewContent("v_nested2 view: ", v_nested2, write_log_fn)

    logViewContent("Nested view does not copy view: ", pointer(v_nested2) == pointer(A), write_log_fn)
end

function test_mixed_indexing2(write_log_fn)
    # -----------------------------
    # Mixed indexing for 3D array
    # -----------------------------
    D = Algebra42.reshape(0:26, (3, 3, 3))

    v_mixed = Algebra42.create_view_from_indices(D, 2, :, 2:3)

    logViewContent("v_mixed view: ", v_mixed, write_log_fn)

    logViewContent("v_mixed size ok: ",  size(v_mixed) == (3, 2), write_log_fn)


    logViewContent("v_mixed is not copy: ",  pointer(v_mixed) == pointer(D), write_log_fn)
end

# function test_boolean_mask(write_log_fn)
#     # -----------------------------
#     # Boolean mask creates copy
#     # -----------------------------
#     mask = B .> 4
#     v_mask = B[mask]
#     @test size(v_mask) == (4,)  # 4 elements greater than 4
#     @test all(v_mask .> 4)
# end


function run_tests_my_view(write_log_fn)
    test_basic_view2(write_log_fn)
    test_colon_view2(write_log_fn)
    test_dimension_droping2(write_log_fn) # scalar
    test_strided_slice2(write_log_fn)
    test_view_of_view2(write_log_fn)
    test_fancy_indexing_copy2(write_log_fn)
    test_scalar_view2(write_log_fn)
    #test_transpose_view2(write_log_fn)
    test_write_through_view2(write_log_fn)
    test_trivial_view2(write_log_fn)
    test_shape_and_strides_consistency2(write_log_fn)
    test_nested_view2(write_log_fn)
    test_mixed_indexing2(write_log_fn)
   # test_boolean_mask(write_log_fn)
end