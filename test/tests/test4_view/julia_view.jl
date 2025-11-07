
function logViewContent(name, data::Bool, log)
    log("\n" * name)
    log(data ? "true" : "false")
end

function logViewContent(name, data, log)
    log("\n" * name)
    write_content(data, log)
end

function get_root_parent(A)
    p = parent(A)
    # Le parent d'un tableau racine est lui-même.
    # On itère jusqu'à ce que le parent soit l'objet actuel.
    return p === A ? A : get_root_parent(p)
end

function shares_memory(A, B)
    # Utilise l'opérateur d'identité '===' pour vérifier si les deux objets
    # racines sont la même instance en mémoire.
    return get_root_parent(A) === get_root_parent(B)
end


bench_view(A, indices...) = @view A[indices...]
bench_fancy1(A, rows, cols) = A[rows, :][:, cols]
bench_fancy2(A, rows, cols) = A[rows, cols]

function test_basic_view(log, bench)
    # -----------------------------
    # Basic contiguous view
    # -----------------------------
    A = reshape(collect(0:15), 4, 4)
    v = @view A[2:3, 2:4]

    logViewContent("basic A", A, log)

    logViewContent("basic view 1", v, log)

    logViewContent("basic view size ok", size(v) == (2, 3), log)
    logViewContent("basic view access 1 ok", v[1, 1] == A[2, 2], log)
    logViewContent("basic view access 2 ok", v[2, 2] == A[3, 3], log)

    v[1, 1] = -99

    logViewContent("basic view 1 (modified)", A[2, 2] == -99, log)
    bench("basic view", bench_view, args=(A, 2:3, 2:4), function_used="view")
end


function test_colon_view(log, bench)
    # -----------------------------
    # Colon view
    # -----------------------------
    B = reshape(collect(0:8), 3, 3)
    v_colon = @view B[:, :]

    logViewContent("colon view 1", v_colon, log)

    logViewContent("colon size ok: ", size(v_colon) == size(B), log)

    logViewContent("same memory ok: ", pointer(v_colon) == pointer(B), log)

    bench("colon view", bench_view, args=(B, :, :), function_used="view")
end

function test_dimension_droping(log, bench)
    # -----------------------------
    # Dimension dropping
    # -----------------------------
    A = reshape(collect(0:15), 4, 4)

    row = @view A[2, :]

    bench("view dimension dropping 1", bench_view, args=(A, 2, :), function_used="view")

    col = @view A[:, 3]

    bench("view dimension dropping 2", bench_view, args=(A, :, 3), function_used="view")

    logViewContent("row view:", row, log)
    logViewContent("col view:", col, log)

    logViewContent("row view size ok:", size(row) == (4,), log)
    logViewContent("col view size ok:", size(col) == (4,), log)

    logViewContent("row view access ok:", row[1] == A[2, 1], log)
    logViewContent("col view access ok:", col[4] == A[4, 3], log)
end

function test_strided_slice(log, bench)

    A = reshape(collect(0:15), 4, 4)

    v_stride = @view A[1:2:4, 1:2:4]

    logViewContent("strided slice view:", v_stride, log)

    logViewContent("strided slice size ok:", size(v_stride) == (2, 2), log)

    logViewContent("strided slice access ok 1:", v_stride[1, 1] == A[1, 1], log)

    logViewContent("strided slice access ok 2:", v_stride[2, 2] == A[3, 3], log)

    logViewContent("strided share memory:", pointer(A) == pointer(v_stride), log)

    bench("view strided slide", bench_view, args=(A, 1:2:4, 1:2:4), function_used="view")
end

function test_view_of_view(log, bench)
    A = reshape(collect(0:15), 4, 4)

    v1 = @view A[2:4, 2:4]

    logViewContent("view of view v1:", v1, log)

    bench("view of view v1", bench_view, args=(A, 2:4, 2:4), function_used="view")

    v2 = @view v1[2:3, 1:2]

    bench("view of view v2", bench_view, args=(v1, 2:3, 1:2), function_used="view")

    logViewContent("view of view v2:", v2, log)

    logViewContent("view v2 size ok:", size(v2) == (2, 2), log)

    logViewContent("view v2 access ok:", v2[1, 1] == A[3, 2], log)

    logViewContent("v2 share memory:", shares_memory(A, v2), log)
end

function test_fancy_indexing_copy(log, bench)
    A = reshape(collect(0:15), 4, 4)
    rows = [1, 3, 4]
    cols = [2, 4]
    fancy = A[rows, :][:, cols]  # creates a copy

    logViewContent("fancy index: ", fancy, log)

    logViewContent("fancy index is copy: ", pointer(A) != pointer(fancy), log)

    logViewContent("fancy index size ok: ", size(fancy) == (3, 2), log)

    logViewContent("fancy index access 1 ok: ", fancy[1, 1] == A[1, 2], log)

    logViewContent("fancy index access 2 ok: ", fancy[3, 2] == A[4, 4], log)

    bench("fancy", bench_fancy1, args=(A, rows, cols), function_used="fancy")
end

function test_scalar_view(log, bench)
    A = reshape(collect(0:8), 3, 3)

    # -----------------------------
    # Scalar view
    # -----------------------------
    s = A[2, 3]

    logViewContent("scalar view: ", s, log)

    logViewContent("scalar content: ", s == 7, log)

    logViewContent("is numeric: ", typeof(s) <: Number, log)
end

function test_transpose_view(log, bench)
    B = reshape(collect(0:8), 3, 3)

    At = transpose(B)

    logViewContent("transpose view: ", At, log)
    logViewContent("transpose is view: ", pointer(A) == pointer(At), log)
    logViewContent("transpose size ok: ", size(At) == (3, 3), log)
    logViewContent("transpose access ok: ", At[1, 2] == B[2, 1], log)
end

function test_write_through_view(log, bench)
    B = reshape(collect(0:8), 3, 3)

    v_write = @view B[2:3, 2:3]

    logViewContent("write through view: ", v_write, log)
    v_write[1, 1] = -42

    logViewContent("write through parent: ", B[2, 2] == -42, log)
end

function test_trivial_view(log, bench)
    A = reshape(collect(0:8), 3, 3)

    v = @view A[:, :]

    logViewContent("trivial view: ", v, log)

    logViewContent("trivial view is same object: ", v === A, log)

    logViewContent("trivial view is parent: ", v === A ||
            (pointer(v) == pointer(A) && strides(v) == strides(A) && size(v) == size(A)), log)

    bench("trivial view", bench_view, args=(A, :, :), function_used="view")
end


function test_shape_and_strides_consistency(log, bench)
    # -----------------------------
    # Shape and stride consistency
    # -----------------------------
    C = reshape(collect(0:11), 3, 4)

    v_stride2 = @view C[1:2:end, :]

    logViewContent("v_stride2 view: ", v_stride2, log)

    logViewContent("v_stride2 size: ", size(v_stride2) == (2, 4), log)

    logViewContent("v_stride2 strides ok: ", stride(v_stride2, 1) == 2 * stride(C, 1), log)

    bench("shape and strides", bench_view, args=(C, 1:2:(size(C, 1)), :), function_used="view")
end

function test_fancy_indexing_diagonal(log, bench)
    C = reshape(collect(0:15), 4, 4)

    v_fancy_diag = C[[1, 2], [3, 4]]

    logViewContent("v_fancy_diag view: ", v_fancy_diag, log)
    logViewContent("v_fancy_diag is copy: ", pointer(C) != pointer(v_fancy_diag), log)
    logViewContent("v_fancy_diag size: ", size(v_fancy_diag) == (2,), log)

    bench("fancy diagonal", bench_fancy2, args=(C, [1, 2], [3, 4]), function_used="fancy")
end

function test_nested_view(log, bench)

    A = reshape(collect(0:15), 4, 4)
    # -----------------------------
    # Nested view does not copy
    # -----------------------------
    v_nested1 = @view A[2:end, :]

    logViewContent("v_nested1 view: ", v_nested1, log)


    v_nested2 = @view v_nested1[:, 3:end]

    logViewContent("v_nested2 view: ", v_nested2, log)

    logViewContent("Nested view does not copy view: ", shares_memory(v_nested2, A), log)
end

function test_mixed_indexing(log, bench)
    # -----------------------------
    # Mixed indexing for 3D array
    # -----------------------------
    D = reshape(collect(0:26), 3, 3, 3)

    v_mixed = @view D[2, :, 2:3]

    logViewContent("v_mixed view: ", v_mixed, log)

    logViewContent("v_mixed size ok: ", size(v_mixed) == (3, 2), log)


    logViewContent("v_mixed is not copy: ", shares_memory(v_mixed, D), log)

    bench("view mixed indexing", bench_view, args=(D, 2, :, 2:3), function_used="view")
end

# function test_boolean_mask(log)
#     # -----------------------------
#     # Boolean mask creates copy
#     # -----------------------------
#     mask = B .> 4
#     v_mask = B[mask]
#     @test size(v_mask) == (4,)  # 4 elements greater than 4
#     @test all(v_mask .> 4)
# end


function test_permutedims_reshape_dropdims(log, bench)

    A = [1.0 2.0; 3.0 4.0]
    B = Array{Float64,3}(Base.reshape(collect(1:24), (4, 3, 2)))
    C = [1.0, 2.0, 3.0, 4.0]
    D = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
    E = Base.reshape(C, (4, 1))
    F = Base.reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (1, 2, 4))
    G = collect(1:24)

    tests = [
        # === permutedims ===
        (
            "permutedims", (a) -> Base.permutedims(a), (A,)
        ),
        (
            "permutedims(3D)", (b) -> Base.permutedims(b, (2, 3, 1)), (B,)
        ),

        # === reshape ===
        (
            "reshape 1", (c) -> Base.reshape(c, (2, 2)), (C,)
        ),
        (
            "reshape 2", (d) -> Base.reshape(d, (2, 2, 2)), (D,)
        ),

        # === dropdims ===
        (
            "dropdims 1", (e) -> Base.dropdims(e, dims=2), (E,)
        ),
        (
            "dropdims 2", (f) -> Base.dropdims(f, dims=1), (F,)
        ),

        # === combo case ===
        (
            "reshape + permutedims",
            () -> Base.permutedims(Base.reshape(Array{Float64,1}(1:6), (2, 3)), (2, 1)),
            ()
        ),
        (
            "permutedims + reshape", (g) ->
            Base.permutedims(Array{Float64,3}(Base.reshape(g, (4, 3, 2))), (2, 3, 1)),
            (G,)
        ),
        (
            "reshape + dropdims",
            () -> Base.dropdims(Base.reshape(collect(1.0:4.0), (1, 4)), dims=1), ()
        )
    ]

    for (name, bench_fn, args) in tests
        bench(name, bench_fn, args=args, function_used="mixed")
        # logResult(name, bench_fn(args...), log)
    end
end


function run_tests_julia_view(log, bench)
    test_basic_view(log, bench)
    test_colon_view(log, bench)
    test_dimension_droping(log, bench) # scalar
    test_strided_slice(log, bench)
    test_view_of_view(log, bench)
    test_fancy_indexing_copy(log, bench)
    test_scalar_view(log, bench)
    #test_transpose_view(log)
    test_write_through_view(log, bench)
    test_trivial_view(log, bench)
    test_shape_and_strides_consistency(log, bench)
    test_fancy_indexing_diagonal(log, bench)
    test_nested_view(log, bench)
    test_mixed_indexing(log, bench)
    # test_boolean_mask(log)
    test_permutedims_reshape_dropdims(log, bench)
end
