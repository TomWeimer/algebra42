using Algebra42
using .TestData
using .TestUtils
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

function logViewContent(name, data::Bool, log)
    log("\n" * name)
    log(data ? "true" : "false")
end

function logViewContent(name, data, log)
    log("\n" * name)
    write_content(data, log)
end

bench_nd_view(A, indices...) = @view A[indices...]
bench_nd_fancy1(A, rows, cols) = A[rows, :][:, cols]
bench_nd_fancy2(A, rows, cols) = A[rows, cols]


function test_basic_view2(log, bench)
    # -----------------------------
    # Basic contiguous view
    # -----------------------------

    A = Algebra42.reshape(collect(0:15), (4, 4))

    logViewContent("basic A", A, log)

    v = A[2:3, 2:4]

    logViewContent("basic view 1", v, log)

    logViewContent("basic view size ok", size(v) == (2, 3), log)
    logViewContent("basic view access 1 ok", v[1, 1] == A[2, 2], log)
    logViewContent("basic view access 2 ok", v[2, 2] == A[3, 3], log)

    v[1, 1] = -99

    logViewContent("basic view 1 (modified)", A[2, 2] == -99, log)

    bench("basic view", bench_nd_view, args=(A, 2:3, 2:4), function_used="view")
end


function test_colon_view2(log, bench)
    # -----------------------------
    # Colon view
    # -----------------------------
    B = Algebra42.reshape(collect(0:8), (3, 3))
    v_colon = B[:, :]

    logViewContent("colon view 1", v_colon, log)

    logViewContent("colon size ok: ", size(v_colon) == size(B), log)

    logViewContent("same memory ok: ", pointer(v_colon) == pointer(B), log)

    bench("colon view", bench_nd_view, args=(B, :, :), function_used="view")
end

function test_dimension_droping2(log, bench)
    # -----------------------------
    # Dimension dropping
    # -----------------------------
    A = Algebra42.reshape(collect(0:15), (4, 4))

    row = A[2, :]

    logViewContent("row view:", row, log)

    bench("view dimension dropping 1", bench_nd_view, args=(A, 2, :), function_used="view")

    col = A[:, 3]

    logViewContent("col view:", col, log)

    logViewContent("row view size ok:", size(row) == (4,), log)
    logViewContent("col view size ok:", size(col) == (4,), log)

    logViewContent("row view access ok:", row[1] == A[2, 1], log)
    logViewContent("col view access ok:", col[4] == A[4, 3], log)

    bench("view dimension dropping 2", bench_nd_view, args=(A, :, 3), function_used="view")
end

function test_strided_slice2(log, bench)

    A = Algebra42.reshape(collect(0:15), (4, 4))

    v_stride = A[1:2:4, 1:2:4]

    logViewContent("strided slice view:", v_stride, log)

    logViewContent("strided slice size ok:", size(v_stride) == (2, 2), log)

    logViewContent("strided slice access ok 1:", v_stride[1, 1] == A[1, 1], log)

    logViewContent("strided slice access ok 2:", v_stride[2, 2] == A[3, 3], log)

    logViewContent("strided share memory:", pointer(A) == pointer(v_stride), log)

    bench("view strided slide", bench_nd_view, args=(A, 1:2:4, 1:2:4), function_used="view")
end

function test_view_of_view2(log, bench)
    A = Algebra42.reshape(collect(0:15), (4, 4))

    v1 = A[2:4, 2:4]

    bench("view of view v1", bench_nd_view, args=(A, 2:4, 2:4), function_used="view")

    logViewContent("view of view v1:", v1, log)

    v2 = v1[2:3, 1:2]

    bench("view of view v2", bench_nd_view, args=(v1, 2:3, 1:2), function_used="view")

    logViewContent("view of view v2:", v2, log)

    logViewContent("view v2 size ok:", size(v2) == (2, 2), log)

    logViewContent("view v2 access ok:", v2[1, 1] == A[3, 2], log)

    logViewContent("v2 share memory:", pointer(A) == pointer(v2), log)
end

function test_fancy_indexing_copy2(log, bench)
    A = Algebra42.reshape(collect(0:15), (4, 4))
    rows = [1, 3, 4]
    cols = [2, 4]
    fancy = A[rows, :][:, cols]  # creates a copy

    logViewContent("fancy index: ", fancy, log)

    logViewContent("fancy index is copy: ", pointer(A) != pointer(fancy), log)

    logViewContent("fancy index size ok: ", size(fancy) == (3, 2), log)

    logViewContent("fancy index access 1 ok: ", fancy[1, 1] == A[1, 2], log)

    logViewContent("fancy index access 2 ok: ", fancy[3, 2] == A[4, 4], log)

    bench("fancy", bench_nd_fancy1, args=(A, rows, cols), function_used="fancy")
end

function test_scalar_view2(log, bench)
    A = Algebra42.reshape(collect(0:8), (3, 3))

    # -----------------------------
    # Scalar view
    # -----------------------------
    s = A[2, 3]

    logViewContent("scalar view: ", s, log)

    logViewContent("scalar content: ", s == 7, log)

    logViewContent("is numeric: ", typeof(s) <: Number, log)
end

function test_transpose_view2(log, bench)
    B = Algebra42.reshape(collect(0:8), (3, 3))

    At = transpose(B)

    logViewContent("transpose view: ", At, log)
    logViewContent("transpose is view: ", pointer(A) == pointer(At), log)
    logViewContent("transpose size ok: ", size(At) == (3, 3), log)
    logViewContent("transpose access ok: ", At[1, 2] == B[2, 1], log)
end

function test_write_through_view2(log, bench)
    B = Algebra42.reshape(collect(0:8), (3, 3))

    v_write = B[2:3, 2:3]

    logViewContent("write through view: ", v_write, log)
    v_write[1, 1] = -42

    logViewContent("write through parent: ", B[2, 2] == -42, log)
end

function test_trivial_view2(log, bench)
    A = Algebra42.reshape(collect(0:8), (3, 3))

    v = A[:, :]

    logViewContent("trivial view: ", v, log)

    logViewContent("trivial view is same object: ", v === A, log)

    logViewContent("trivial view is parent: ", v === A ||
            (pointer(v) == pointer(A) && strides(v) == strides(A) && size(v) == size(A)), log)

    bench("trivial view", bench_nd_view, args=(A, :, :), function_used="view")
end


function test_shape_and_strides_consistency2(log, bench)
    # -----------------------------
    # Shape and stride consistency
    # -----------------------------
    C = Algebra42.reshape(collect(0:11), (3, 4))

    v_stride2 = C[1:2:end, :]

    logViewContent("v_stride2 view: ", v_stride2, log)

    logViewContent("v_stride2 size: ", size(v_stride2) == (2, 4), log)

    logViewContent("v_stride2 strides ok: ", stride(v_stride2, 1) == 2 * stride(C, 1), log)

    bench("shape and strides", bench_nd_view, args=(C, 1:2:(size(C, 1)), :), function_used="view")
end

function test_fancy_indexing_diagonal2(log, bench)
    C = Algebra42.reshape(collect(0:15), (4, 4))
    row = NDArray{Int}([[1], [2]])

    v_fancy_diag = C[row, [3, 4]]

    logViewContent("v_fancy_diag view: ", v_fancy_diag, log)
    logViewContent("v_fancy_diag is copy: ", pointer(C) != pointer(v_fancy_diag), log)
    logViewContent("v_fancy_diag size: ", size(v_fancy_diag) == (2,), log)

    bench("fancy diagonal", bench_nd_fancy2, args=(C, row, [3, 4]), function_used="fancy")
end

function test_nested_view2(log, bench)

    A = Algebra42.reshape(collect(0:15), (4, 4))
    # -----------------------------
    # Nested view does not copy
    # -----------------------------
    v_nested1 = A[2:end, :]

    logViewContent("v_nested1 view: ", v_nested1, log)


    v_nested2 = v_nested1[:, 3:end]

    logViewContent("v_nested2 view: ", v_nested2, log)

    logViewContent("Nested view does not copy view: ", pointer(v_nested2) == pointer(A), log)
end

function test_mixed_indexing2(log, bench)
    # -----------------------------
    # Mixed indexing for 3D array
    # -----------------------------
    D = Algebra42.reshape(collect(0:26), (3, 3, 3))

    v_mixed = D[2, :, 2:3]


    logViewContent("v_mixed view: ", v_mixed, log)

    logViewContent("v_mixed size ok: ", size(v_mixed) == (3, 2), log)


    logViewContent("v_mixed is not copy: ", pointer(v_mixed) == pointer(D), log)

    bench("view mixed indexing", bench_nd_view, args=(D, 2, :, 2:3), function_used="view")
end

# function test_boolean_mask(log)
#     # -----------------------------
#     # Boolean mask creates copy
#     # -----------------------------
#     mask = B .> 4
#     v_mask = B[mask]
#     @assert size(v_mask) == (4,)  # 4 elements greater than 4
#     @assert all(v_mask .> 4)
# end


# -----------------------------------------------------------------------------
# LES TESTS
# -----------------------------------------------------------------------------
struct MockArray{T,N} <: AbstractArray{T,N}
    data::Base.Vector{T}
    dims::Dims{N}
end

Base.strides(A::MockArray{T,N}) where {T,N} = Algebra42.compute_strides(size(A))

MockArray(data::Base.Vector{T}, dims::Dims{N}) where {T,N} = MockArray{T,N}(data, dims)
Base.size(A::MockArray) = A.dims
Base.axes(A::MockArray) = map(Base.OneTo, A.dims)

_checkcontiguous(::Type{Bool}, A::MockArray) = false

function Base.getindex(P::MockArray{T,N}, I::Vararg{Int,N}) where {T,N}
    # Convert Cartesian index I to linear index for the column-major MockArray
    # Assuming N=2: I = (row, col)
    row, col = I[1], I[2]
    rows = size(P)[1]

    # Calculate column-major linear index: (col - 1) * rows + row
    lin_idx = (col - 1) * rows + row

    return P.data[lin_idx]
end


function test_reshape()

    # --- Préparation des données parentes ---
    # Parent 3x4x5 avec des données séquentielles
    parent_data = Base.reshape(collect(1:60), 3, 4, 5)

    function test_reshape1(parent_data)
        P = parent_data # 3x4x5
        A = Algebra42.reshape(P, (10, 6)) # Vue 10x6

        @assert size(A) == (10, 6)
        @assert length(A) == 60

        # Test de lecture (lecture de A[i, j] doit donner P[x, y, z])
        # A[1, 1] -> Index Linéaire 1 -> P[1, 1, 1] = 1
        @assert A[1, 1] == 1

        # A[10, 1] -> Index Linéaire 10 -> P[1, 4, 1] (10 = 1 + 3*3 + 12*0)
        # 1 + 3*(4-1) + 12*(1-1) = 10
        @assert A[10, 1] == P[1, 4, 1]

        # A[1, 2] -> Index Linéaire 11 -> P[2, 4, 1]
        # 2 + 3*(4-1) + 12*(1-1) = 11
        @assert A[11] == P[2, 4, 1]

        # A[10, 6] -> Index Linéaire 60 -> P[3, 4, 5] = 60
        @assert A[10, 6] == 60

        # Test d'écriture
        setindex!(A, 999, 5, 3)
        # A[5, 3] est l'index linéaire 25.
        # Index 25 -> P[1, 1, 3] (1 + 3*0 + 12*2 = 25)
        @assert P[1, 1, 3] == 999
        @assert A[5, 3] == 999
    end

    function test_reshape2()
        # Parent 60 éléments, IndexLinear
        P_linear = collect(101:160) # Vector{Int}
        A_linear = Algebra42.reshape(P_linear, (5, 12)) # Vue 5x12

        @assert size(A_linear) == (5, 12)
        @assert Base.IndexStyle(A_linear) == IndexLinear() # Doit être un ReshapedArray42LF (sans MI)

        # A[1, 1] -> Index Linéaire 1 -> P[1]
        @assert A_linear[1, 1] == 101

        # A[5, 1] -> Index Linéaire 5 -> P[5]
        @assert A_linear[5, 1] == 105

        # A[1, 2] -> Index Linéaire 6 -> P[6]
        @assert A_linear[1, 2] == 106

        # A[5, 12] -> Index Linéaire 60 -> P[60]
        @assert A_linear[5, 12] == 160

        # Test d'écriture
        setindex!(A_linear, 777, 3, 7)
        # A[3, 7] est l'index linéaire (3 + 5*(7-1)) = 33.
        @assert P_linear[33] == 777
        @assert A_linear[3, 7] == 777
    end

    function test_reshape3()
        # 1. Define a parent array P (simulating a 3x4 array)
        P = MockArray(collect(1:12), (3, 4)) # Parent is 3x4

        # 2. Create a ReshapedArray42 (e.g., reshape 3x4 into a 12-element Vector)
        R = ReshapedArray42(P, (12,), Val(false))

        result = getindex(R, 7)
        @assert result == 7

        # R[10] should map to P[1, 4] which holds value 10
        result_2 = getindex(R, 10)

        @assert result_2 == 10
    end

    try
        test_reshape1(parent_data)
        test_reshape2()
        test_reshape3()
    catch e
        if (e isa AssertionError)
            log("test_reshape: failed")
        else
            rethrow(e)
        end
    end
end

function test_nd_permutedims_reshape_dropdims(log, bench)

    A = to_ndarray([1.0 2.0; 3.0 4.0])
    B = NDArray{Float64}(Algebra42.reshape(to_ndarray(collect(1:24)), (4, 3, 2)))
    C = to_ndarray([1.0, 2.0, 3.0, 4.0])
    D = to_ndarray([1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0])
    E = Algebra42.reshape(C, (4, 1))
    F = Algebra42.reshape(to_ndarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), (1, 2, 4))
    G = to_ndarray(collect(1:24))

    tests = [
        # === permutedims ===
        (
            "permutedims", (a) -> Algebra42.permutedims(a), (A,)
        ),
        (
            "permutedims(3D)", (b) -> Algebra42.permutedims(b, (2, 3, 1)), (B,)
        ),

        # === reshape ===
        (
            "reshape 1", (c) -> Algebra42.reshape(c, (2, 2)), (C,)
        ),
        (
            "reshape 2", (d) -> Algebra42.reshape(d, (2, 2, 2)), (D,)
        ),

        # === dropdims ===
        (
            "dropdims 1", (e) -> Algebra42.dropdims(e, dims=2), (E,)
        ),
        (
            "dropdims 2", (f) -> Algebra42.dropdims(f, dims=1), (F,)
        ),

        # === combo case ===
        (
             "reshape + permutedims", 
            () -> Algebra42.permutedims(Algebra42.reshape(NDArray{Float64}(1:6), (2, 3)), (2, 1)),
            ()
        ),
        (
            "permutedims + reshape", (g) ->
             Algebra42.permutedims(NDArray{Float64}(Algebra42.reshape(g, (4, 3, 2))), (2, 3, 1)), 
             (G,)
        ),
        (
           "reshape + dropdims",
            () -> Algebra42.dropdims(Algebra42.reshape(NDArray{Float64}(1:4), (1, 4)), dims=1), ()
        )
    ]

    for (name, nd_bench_fn, args) in tests
        bench(name, nd_bench_fn, args=args, function_used="mixed")
        #logResult(name, nd_bench_fn(args...), log)
    end
end


function run_tests_my_view(log, bench)
    test_basic_view2(log, bench)
    test_colon_view2(log, bench)
    test_dimension_droping2(log, bench) # scalar
    test_strided_slice2(log, bench)
    test_view_of_view2(log, bench)
    test_fancy_indexing_copy2(log, bench)
    test_scalar_view2(log, bench)
    #test_transpose_view2(log)
    test_write_through_view2(log, bench)
    test_trivial_view2(log, bench)
    test_shape_and_strides_consistency2(log, bench)
    test_fancy_indexing_diagonal2(log, bench)
    test_nested_view2(log, bench)
    test_mixed_indexing2(log, bench)
    test_reshape()

    test_nd_permutedims_reshape_dropdims(log, bench)
    # test_boolean_mask(log)
end