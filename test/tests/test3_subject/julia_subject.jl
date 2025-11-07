using .TestData
using .TestUtils

using GeometryBasics
using LinearAlgebra
using RowEchelon
using Debugger

# ──── log ─────────────────────────────────────────────────────────────────────────────────────── #
            
function logResult(name, result::String, log)
    log("\n" * name)
    log(result)
end

function logResult(name, result, log)
    log("\n" * name)
    write_content(result, log)
end

# ──── add ─────────────────────────────────────────────────────────────────────────────────────── #
            
function test_add_function(log, bench)

    logAddResult(name, v1, v2, log) = (
        logResult(name, v1 .+ v2, log)
    )

    benchAdd(v1, v2) = v1 .+ v2

    add_tests = [
        ("vector + vector 1", vector_3.matrix, vector_4.matrix),
        ("vector + vector 2", vector_3.matrix, vector_3.matrix),
    ]

    for (name, v1, v2) in add_tests
        logAddResult(name, v1, v2, log)
        bench(name, benchAdd, args=(v1, v2), function_used="add")
    end
end

# ──── sub ─────────────────────────────────────────────────────────────────────────────────────── #
            
function test_sub_function(log, bench)

    logSubResult(name, elem1, elem2, log) = (
        logResult(name, elem1 .- elem2, log)
    )

    benchSub(v1, v2) = v1 .- v2

    sub_tests = [
        ("vector - vector 1", vector_3.matrix, vector_4.matrix),
        ("vector - vector 2", vector_3.matrix, vector_3.matrix),
    ]

    for (name, v1, v2) in sub_tests
        logSubResult(name, v1, v2, log)
        bench(name, benchSub, args=(v1, v2), function_used="sub")
    end
end

# ──── prod ────────────────────────────────────────────────────────────────────────────────────── #
            
function test_prod_function(log, bench)

    logProdResult(name, a, b, log) = (
        logResult(name, a .* b, log)
    )

    benchProd(a, b) = a .* b

    prod_tests = [
        ("vector * 42 1", vector_3.matrix, scalar.data),
        ("vector * 42 2", vector_4.matrix, scalar.data),
    ]

    for (name, a, b) in prod_tests
        logProdResult(name, a, b, log)
        bench(name, benchProd, args=(a, b), function_used="prod")
    end

end

# ──── linear combination ──────────────────────────────────────────────────────────────────────── #
            
function test_linear_combination(log, bench)

    linear_combinations = [
        ("linear comb 1:", [Base.Matrix{Float64}(I, 3, 3)[:, i] for i in 1:3], [10., -2., 0.5]),
        ("linear comb 2:", [[1., 2., 3.], [0., 10., -100.]], [10., -2.]),
    ]

    logLinearCombination(name, coefs, vectors, log) = (
        logResult(name, sum(a .* v for (a, v) in zip(coefs, vectors)), log)
    )

    bench_linear_comb(coefs, vectors) = (
        sum(a .* v for (a, v) in zip(coefs, vectors))
    )

    for (name, vectors, coefs) in linear_combinations
        logLinearCombination(name, coefs, vectors, log)
        bench(name, bench_linear_comb, args=(coefs, vectors), function_used="linear-comb")
    end
end

# ──── linear interpolation ────────────────────────────────────────────────────────────────────── #
            
function test_lerp(log, bench)

    lerp_expected_values = [
        0.0,
        1.0,
        0.5,
        27.3,
        [2.6, 1.3],
        [11.0 5.5; 16.5 22.0]
    ]

    # Loop over them
    for (i, val) in enumerate(lerp_expected_values)
        logResult("lerp $i:", val, log)
    end
end

# ──── dot product ─────────────────────────────────────────────────────────────────────────────── #
            
function test_dot(log, bench)

    dot_tests = [
        ("dot 1:", [0., 0.], [1., 1.]),
        ("dot 2:", [1., 1.], [1., 1.]),
        ("dot 3:", [-1., 6.], [3., 2.]),
    ]

    logDot(name, u, v) = logResult(name, LinearAlgebra.dot(u, v), log)

    benchDot(u, v) = LinearAlgebra.dot(u, v)

    for (name, u, v) in dot_tests
        logDot(name, u, v)
        bench(name, benchDot, args=(u, v), function_used="dot")
    end
end

# ──── norms ───────────────────────────────────────────────────────────────────────────────────── #

function test_norm(log, bench)
    norm_tests = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, -2.0]
    ]

    bench_norm1(u) = LinearAlgebra.norm(u, 1)
    bench_norm(u) = LinearAlgebra.norm(u)
    bench_norminf(u) = LinearAlgebra.norm(u, Inf)

    log_norms(name, u) = logResult(name, 
    "$(LinearAlgebra.norm(u, 1)), $(LinearAlgebra.norm(u)), $(LinearAlgebra.norm(u, Inf))", log)

    for (i, u) in enumerate(norm_tests)
        log_norms("norms $i:", u)
        bench("norm1 $i", bench_norm1, args=(u,), function_used="norm1")
        bench("norm $i",  bench_norm, args=(u,), function_used="norm")
        bench("norm_inf $i", bench_norminf, args=(u,), function_used="norm_inf")
    end
end

# ──── angle between vectors ───────────────────────────────────────────────────────────────────── #
            
function test_angle_cos(log, bench)

    angle_tests = (
        ("angle cos 1:", [1.0, 0.0], [1.0, 0.0]),
        ("angle cos 2:", [1.0, 0.0], [0.0, 1.0]),
        ("angle cos 3:", [-1.0, 1.0], [1.0, -1.0]),
        ("angle cos 4:", [2.0, 1.0], [4.0, 2.0]),
        ("angle cos 5:", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    )

    angle_vector(a, b) = LinearAlgebra.dot(a, b) / (LinearAlgebra.norm(a) * LinearAlgebra.norm(b))


    logAngle(name, a, b) = logResult(name, angle_vector(a, b), log)

    for (name, u, v) in angle_tests
        logAngle(name, u, v)
        bench(name, angle_vector, args=(u, v), function_used="angle")
    end

end

# ──── cross product ───────────────────────────────────────────────────────────────────────────── #
            

function test_cross_product(log, bench)

    cross_tests = [
        ("cross product 1:", [0., 0., 1.], [1., 0., 0.]),
        ("cross product 2:", [1., 2., 3.], [4., 5., 6.]),
        ("cross product 3:", [4., 2., -3.], [-2., -5., 16.])
    ]

    logCross(name, u, v) = logResult(name, LinearAlgebra.cross(u, v), log)

    benchCross(u, v) = LinearAlgebra.cross(u, v)

    for (name, u, v) in cross_tests
        logCross(name, u, v)
        bench(name, benchCross, args=(u, v), function_used="cross")
    end
end


# ──── matrix multiplication ───────────────────────────────────────────────────────────────────── #
            
function test_mul(log, bench)

    testsMult = (
        ("mul 1:", [1.0 0.0; 0.0 1.0], [4.0, 2.0]),
        ("mul 2:", [2.0 0.0; 0.0 2.0], [4.0, 2.0]),
        ("mul 3:", [2.0 -2.0; -2.0 2.0], [4.0, 2.0]),
        ("mul 4:", [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]),
        ("mul 5:", [1.0 0.0; 0.0 1.0], [2.0 1.0; 4.0 2.0]),
        ("mul 6:", [3.0 -5.0; 6.0 8.0], [2.0 1.0; 4.0 2.0])
    )

    logMul(name, A, b) = logResult(name, A * b, log)
    benchMul(A, b) = A * b

    for (name, A, b) in testsMult
        logMul(name, A, b)
        bench(name, benchMul, args=(A, b), function_used="mat_mul")
    end
end

# ──── trace ───────────────────────────────────────────────────────────────────────────────────── #
            
function test_trace(log, bench)

    trace_tests = [
        ("trace 1:", [1. 0.; 0. 1.]),
        ("trace 2:", [2. -5. 0.; 4. 3. 7.; -2. 3. 4.]),
        ("trace 3:", [-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])
    ]

    bench_trace(A) = LinearAlgebra.tr(A)

    logTrace(name, A) = logResult(name, LinearAlgebra.tr(A), log)

    for (name, A) in trace_tests
        logTrace(name, A)
        bench(name, bench_trace, args=(A,), function_used="trace")
    end
end

# ──── transpose ───────────────────────────────────────────────────────────────────────────────── #
            
function test_transpose(log, bench)

    transpose_cases = [
        ("transpose 1:", [1.0 2.0; 3.0 4.0]),
        ("transpose 2:", [2.0 -5.0 0.0; 4.0 3.0 7.0; -2.0 3.0 4.0]),
        ("transpose 3:", [-2.0 -8.0 4.0; 1.0 -23.0 4.0; 0.0 6.0 4.0]),
        ("transpose 4:", [1.0 2.0; 3.0 4.0; 5.0 6.0])
    ]

    logTranspose(name, A) = logResult(name, LinearAlgebra.transpose(A), log)

    benchTranspose(A) = LinearAlgebra.transpose(A)

    for (name, A) in transpose_cases
        logTranspose(name, A)
        bench(name, benchTranspose, args=(A,), function_used="transpose")
    end
end

# ──── reduced row echelon form ────────────────────────────────────────────────────────────────── #

function test_row_echelon(log, bench)
    row_echelon_tests = [
        ("row_echelon 1:", [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]),
        ("row_echelon 2:", [1. 2.; 3. 4.]),
        ("row_echelon 3:", [1. 2.; 2. 4.]),
        ("row_echelon 4:", [8. 5. -2. 4. 28.; 4. 2.5 20. 4. -4.; 8. 5. 1. 4. 17.])
    ]

    logRowEch(name, A) = logResult(name, RowEchelon.rref(A), log)

    bench_rref(A) = RowEchelon.rref(A)

    for (name, A) in row_echelon_tests
        logRowEch(name, A)
        bench(name, bench_rref, args=(A,), function_used="rref")
    end
end

# ──── determinant ─────────────────────────────────────────────────────────────────────────────── #


function test_determinant(log, bench)
    # Define test matrices
    determinant_cases = [
        ("determinant 1:", [1.0 -1.0; -1.0 1.0]),
        ("determinant 2:", [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]),
        ("determinant 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0]),
        ("determinant 4:", [8.0 5.0 -2.0 4.0; 4.0 2.5 20.0 4.0; 8.0 5.0 1.0 4.0; 28.0 -4.0 17.0 1.0]
        )
    ]

    logDet(name, A) = logResult(name, LinearAlgebra.det(A), log)
    benchDet(A)     = LinearAlgebra.det(A)

    for (name, A) in determinant_cases
        logDet(name, A)
        bench(name, benchDet, args=(A,), function_used="det")
    end
end

# ──── inverse ─────────────────────────────────────────────────────────────────────────────────── #
            
function test_inverse(log, bench)

    inverse_cases = [
        ("inverse 1:", [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
        ("inverse 2:", [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]),
        ("inverse 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0])
    ]

    logInverse(name, A) = logResult(name, LinearAlgebra.inv(A), log)

    benchInverse(A) = LinearAlgebra.inv(A)

    for (name, A) in inverse_cases
        logInverse(name, A)
        bench(name, benchInverse, args=(A,), function_used="inverse")
    end

end

# ──── rank ────────────────────────────────────────────────────────────────────────────────────── #

function test_rank(log, bench)
    
    rank_cases = [
        ("rank 1:", [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
        ("rank 2:", [1.0 2.0 0.0 0.0; 2.0 4.0 0.0 0.0; -1.0 2.0 1.0 1.0]),
        ("rank 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0; 21.0 18.0 7.0])
    ]

    logRank(name, A) = logResult(name, LinearAlgebra.rank(A), log)
    
    benchRank(A) = LinearAlgebra.rank(A)

    for (name, A) in rank_cases
        logRank(name, A)
        bench(name, benchRank, args=(A,),  function_used="rank")
    end
end

# ──── run tests ───────────────────────────────────────────────────────────────────────────────── #
            
function run_tests_julia_subject(log, bench)

    test_functions = [
        test_add_function,
        test_sub_function,
        test_prod_function, # scalar
        test_linear_combination,
        test_lerp,
        test_dot,
        test_norm,
        test_angle_cos,
        test_cross_product,
        test_mul,
        test_trace,
        test_transpose,
        test_row_echelon,
        test_determinant,
        test_inverse,
        test_rank
    ]

    for test_function in test_functions
        test_function(log, bench)
    end
end