using Algebra42
using .TestData
using .TestUtils

using Debugger

to_ndarray(A::Number) = A
to_ndarray(A::TestDataScalar) = A.data
to_ndarray(A::NDArrayData) = to_ndarray(A.matrix)
to_ndarray(A::Algebra42.AbstractNDArray) = A
to_ndarray(A::AbstractMatrix) = Algebra42.Matrix{eltype(A)}(A)
to_ndarray(A::AbstractVector) = Algebra42.Vector{eltype(A)}(A)

to_ndarrays() = ()

function to_ndarrays(A, arrays...)
    return (to_ndarray(A), to_ndarrays(arrays...)...)
end


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

function test_nd_add_function(log, bench)

    # function test_add_function(log)
    #     println(stderr, "add function")
    #     logAddResult("vector + vector 1", NDVector{Int}(vector_3.array), NDVector{dtype}(vector_4.array), log)
    #     logAddResult("vector + vector 2", NDVector{dtype}(vector_3.array), NDVector{dtype}(vector_3.array), log)
    #     mustThrowAdd(NDVector{dtype}(vector_4.array), NDVector{dtype}(vector_5.array))
    #     mustThrowAdd(NDVector{dtype}(vector_3.array), NDVector{dtype}(vector_5.array))
    #     mustThrowAdd(NDVector{dtype}(vector_1.array), NDVector{dtype}(vector_4.array))
    #     mustThrowAdd(NDVector{dtype}(vector_4.array), NDVector{dtype}(vector_1.array))
    # end

    logAddResultND(name, elem1, elem2, log) = (
        logResult(name, Algebra42.add(elem1, elem2), log)
    )

    benchNDAdd(v1, v2) = Algebra42.add(v1, v2)

    add_tests = [
        ("vector + vector 1", vector_3, vector_4),
        ("vector + vector 2", vector_3, vector_3),
    ]

    for (name, u, v) in add_tests
        v1, v2 = to_ndarrays(u, v)
        logAddResultND(name, v1, v2, log)
        bench(name, benchNDAdd, args=(v1, v2), function_used="add")
    end

    add_exceptions = [
        (vector_4, vector_5),
        (vector_3, vector_5),
        (vector_1, vector_4),
        (vector_4, vector_1)
    ]

    function mustThrowAdd(elem1, elem2)
        f = () -> Algebra42.add(elem1, elem2)
        @assert exceptionPassed(f, 1, DomainError)
    end


    for (u, v) in add_exceptions
        v1, v2 = to_ndarrays(u, v)
        mustThrowAdd(v1, v2)
    end
end

# ──── sub ─────────────────────────────────────────────────────────────────────────────────────── #

function test_nd_sub_function(log, bench)

    # function test_sub_function(dtype::Type, log)
    #     println(stderr, "sub function")
    #     logSubResult("vector - vector 1", NDVector{dtype}(vector_3.array), NDVector{dtype}(vector_4.array), log)
    #     logSubResult("vector - vector 2", NDVector{dtype}(vector_3.array), NDVector{dtype}(vector_3.array), log)
    #     mustThrowSub(NDVector{dtype}(vector_4.array), NDVector{dtype}(vector_5.array))
    #     mustThrowSub(NDVector{dtype}(vector_3.array), NDVector{dtype}(vector_5.array))
    #     mustThrowSub(NDVector{dtype}(vector_1.array), NDVector{dtype}(vector_4.array))
    #     mustThrowSub(NDVector{dtype}(vector_4.array), NDVector{dtype}(vector_1.array))
    # end

    logSubResultND(name, v1, v2, log) = (
        logResult(name, Algebra42.sub(v1, v2), log)
    )

    benchNDSub(v1, v2) = Algebra42.sub(v1, v2)

    sub_tests = [
        ("vector - vector 1", vector_3, vector_4),
        ("vector - vector 2", vector_3, vector_3),
    ]

    for (name, u, v) in sub_tests
        v1, v2 = to_ndarrays(u, v)
        logSubResultND(name, v1, v2, log)
        bench(name, benchNDSub, args=(v1, v2), function_used="sub")
    end

    sub_exceptions = [
        (vector_4, vector_5),
        (vector_3, vector_5),
        (vector_1, vector_4),
        (vector_4, vector_1)
    ]

    function mustThrowSub(elem1, elem2)
        f = () -> Algebra42.sub(elem1, elem2)
        @assert exceptionPassed(f, 1, DomainError)
    end


    for (u, v) in sub_exceptions
        v1, v2 = to_ndarrays(u, v)
        mustThrowSub(v1, v2)
    end
end

# ──── prod ────────────────────────────────────────────────────────────────────────────────────── #

function test_nd_prod_function(log, bench)

    # scalar
    # function test_prod_function(dtype::Type, log)
    #     println(stderr, "prod function")
    #     logProdResult("vector * 42 1", NDVector{dtype}(vector_3.array), scalar.data, log)
    #     logProdResult("vector * 42 2", NDVector{dtype}(vector_4.array), scalar.data, log)
    #     mustThrowProd(NDVector{dtype}(vector_4.array), scalar.data)
    # end

    function mustThrowProd(elem1, scalar)
        f = () -> Algebra42.prod(scalar, elem1)
        @assert exceptionPassed(f, 1, MethodError)
    end

    logProdResultND(name, elem1, scalar::Number, log) = (
        logResult(name, Algebra42.prod(elem1, scalar), log)
    )

    benchNDProd(a, b) = Algebra42.prod(a, b)

    prod_tests = [
        ("vector * 42 1", vector_3, scalar),
        ("vector * 42 2", vector_4, scalar),
    ]

    for (name, u, s) in prod_tests
        a, b = to_ndarrays(u, s)
        logProdResultND(name, a, b, log)
        bench(name, benchNDProd, args=(a, b), function_used="prod")
    end

    mustThrowProd(to_ndarray(vector_4), scalar.data)
end


# ──── linear combination ──────────────────────────────────────────────────────────────────────── #

function test_nd_linear_combination(log, bench)

    # scalar
    # function test_linear_combination(dtype::Type, log)
    #     println(stderr, "linear_combination function")
    #     e1 = NDVector{Float64}([1., 0., 0.])
    #     e2 = NDVector{Float64}([0., 1., 0.])
    #     e3 = NDVector{Float64}([0., 0., 1.])
    #     coefs = [10., -2., 0.5]
    #     logResult("linear comb 1:", Algebra42.linear_combination([e1, e2, e3], coefs), log)

    #     v1 = NDVector{Float64}([1., 2., 3.])
    #     v2 = NDVector{Float64}([0., 10., -100.])
    #     coefs = [10., -2.]
    #     logResult("linear comb 2:", Algebra42.linear_combination([v1, v2], coefs), log)
    # end


    linear_combinations = [
        ("linear comb 1:", [Base.Matrix{Float64}(I, 3, 3)[:, i] for i in 1:3], [10., -2., 0.5]),
        ("linear comb 2:", [[1., 2., 3.], [0., 10., -100.]], [10., -2.]),
    ]

    logLinearCombinationND(name, coefs, vectors, log) = (
        logResult(name, Algebra42.linear_combination(vectors, coefs), log)
    )

    bench_nd_linear_comb(coefs, vectors) = (
        Algebra42.linear_combination(vectors, coefs)
    )

    for (name, v, coefs) in linear_combinations
        vectors = [to_ndarrays(v...)...]
        logLinearCombinationND(name, coefs, vectors, log)
        bench(name, bench_nd_linear_comb, args=(coefs, vectors), function_used="linear-comb")
    end

end


# ──── linear interpolation ────────────────────────────────────────────────────────────────────── #

function test_nd_lerp(log, bench)

    #   function test_lerp(dtype::Type, log)
    #     logResult("lerp 1:", Algebra42.lerp(0., 1., 0.), log)
    #     logResult("lerp 2:", Algebra42.lerp(0., 1., 1.), log)
    #     logResult("lerp 3:", Algebra42.lerp(0., 1., 0.5), log)
    #     logResult("lerp 4:", Algebra42.lerp(21., 42., 0.3), log)


    #     u = NDVector{Float64}([2., 1.])
    #     v = NDVector{Float64}([4., 2.])

    #     logResult("lerp 5:", Algebra42.lerp(u, v, 0.3), log)

    #     A = NDMatrix{Float64}([2. 1.; 3. 4.])
    #     B = NDMatrix{Float64}([20. 10.; 30. 40.])

    #     logResult("lerp 6:", Algebra42.lerp(A, B, 0.5), log)
    # end


    lerp_cases = [
        ("lerp 1:", (0., 1.), 0.),
        ("lerp 2:", (0., 1.), 1.),
        ("lerp 3:", (0., 1.), 0.5),
        ("lerp 4:", (21., 42.), 0.3),
        ("lerp 5:", ([2., 1.], [4., 2.]), 0.3),
        ("lerp 6:", ([2. 1.; 3. 4.], [20. 10.; 30. 40.]), 0.5)
    ]

    logLerp(name, v1, v2, coef) = logResult(name, Algebra42.lerp(v1, v2, coef), log)

    for (name, vectors, coef) in lerp_cases
        v1, v2 = to_ndarrays(vectors...)
        logLerp(name, v1, v2, coef)
    end
end

# ──── dot product ─────────────────────────────────────────────────────────────────────────────── #

function test_nd_dot(log, bench)

    # function test_dot(dtype::Type, log)

    #     u = NDVector{Float64}([0., 0.])
    #     v = NDVector{Float64}([1., 1.])

    #     logResult("dot 1:", Algebra42.dot(u, v), log)

    #     u = NDVector{Float64}([1., 1.])
    #     v = NDVector{Float64}([1., 1.])

    #     logResult("dot 2:", Algebra42.dot(u, v), log)

    #     u = NDVector{Float64}([-1., 6.])
    #     v = NDVector{Float64}([3., 2.])

    #     logResult("dot 3:", Algebra42.dot(u, v), log)
    # end

    dot_tests = [
        ("dot 1:", [0., 0.], [1., 1.]),
        ("dot 2:", [1., 1.], [1., 1.]),
        ("dot 3:", [-1., 6.], [3., 2.]),
    ]

    logDotND(name, u, v) = logResult(name, Algebra42.dot(u, v), log)

    benchNDDot(u, v) = Algebra42.dot(u, v)

    for (name, a, b) in dot_tests
        u, v = to_ndarrays(a, b)
        logDotND(name, u, v)
        bench(name, benchNDDot, args=(u, v), function_used="dot")
    end
end


# ──── norms ───────────────────────────────────────────────────────────────────────────────────── #

function test_nd_norm(log, bench)

    #     function test_norm(dtype::Type, log)
    #     u = NDVector{Float64}([0., 0., 0.])

    #     logResult("norms 1:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", log)

    #     u = NDVector{Float64}([1., 2., 3.])

    #     logResult("norms 2:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", log)

    #     u = NDVector{Float64}([-1., -2.])

    #     logResult("norms 3:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", log)
    # end

    norm_tests = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, -2.0]
    ]

    bench_nd_norm1(u) = Algebra42.norm_1(u)
    bench_nd_norm(u) = Algebra42.norm(u)
    bench_nd_norminf(u) = Algebra42.norm_inf(u)

    log_nd_norms(name, u) = logResult(name,
        "$(Algebra42.norm_1(u)), $(Algebra42.norm(u)), $(Algebra42.norm_inf(u))", log)

    for (i, a) in enumerate(norm_tests)
        u = to_ndarray(a)
        log_nd_norms("norms $i:", u)
        bench("norm1 $i", bench_nd_norm1, args=(u,), function_used="norm1")
        bench("norm $i", bench_nd_norm, args=(u,), function_used="norm")
        bench("norm_inf $i", bench_nd_norminf, args=(u,), function_used="norm_inf")
    end
end



# ──── angle between vectors ───────────────────────────────────────────────────────────────────── #

function test_nd_angle_cos(log, bench)

    # function test_angle_cos(dtype::Type, log)
    #     u = NDVector{Float64}([1., 0.])
    #     v = NDVector{Float64}([1., 0.])

    #     logResult("angle cos 1:", Algebra42.angle_cos(u, v), log)

    #     u = NDVector{Float64}([1., 0.])
    #     v = NDVector{Float64}([0., 1.])

    #     logResult("angle cos 2:", Algebra42.angle_cos(u, v), log)

    #     u = NDVector{Float64}([-1., 1.])
    #     v = NDVector{Float64}([1., -1.])

    #     logResult("angle cos 3:", Algebra42.angle_cos(u, v), log)

    #     u = NDVector{Float64}([2., 1.])
    #     v = NDVector{Float64}([4., 2.])

    #     logResult("angle cos 4:", Algebra42.angle_cos(u, v), log)

    #     u = NDVector{Float64}([1., 2., 3.])
    #     v = NDVector{Float64}([4., 5., 6.])

    #     logResult("angle cos 5:", Algebra42.angle_cos(u, v), log)
    # end

    angle_tests = (
        ("angle cos 1:", [1.0, 0.0], [1.0, 0.0]),
        ("angle cos 2:", [1.0, 0.0], [0.0, 1.0]),
        ("angle cos 3:", [-1.0, 1.0], [1.0, -1.0]),
        ("angle cos 4:", [2.0, 1.0], [4.0, 2.0]),
        ("angle cos 5:", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    )

    logNDAngle(name, a, b) = logResult(name, Algebra42.angle_cos(a, b), log)
    benchNDAngle(u, v) = Algebra42.angle_cos(u, v)

    for (name, a, b) in angle_tests
        u, v = to_ndarrays(a, b)
        logNDAngle(name, u, v)
        bench(name, benchNDAngle, args=(u, v), function_used="angle")
    end



end

# ──── cross product ───────────────────────────────────────────────────────────────────────────── #


function test_nd_cross_product(log, bench)

    #     function test_cross_product(dtype::Type, log)
    #     u = NDVector{Float64}([0., 0., 1.])
    #     v = NDVector{Float64}([1., 0., 0.])

    #     logResult("cross product 1:", Algebra42.cross_product(u, v), log)

    #     u = NDVector{Float64}([1., 2., 3.])
    #     v = NDVector{Float64}([4., 5., 6.])

    #     logResult("cross product 2:", Algebra42.cross_product(u, v), log)

    #     u = NDVector{Float64}([4., 2., -3.])
    #     v = NDVector{Float64}([-2., -5., 16.])

    #     logResult("cross product 3:", Algebra42.cross_product(u, v), log)
    # end


    cross_tests = [
        ("cross product 1:", [0., 0., 1.], [1., 0., 0.]),
        ("cross product 2:", [1., 2., 3.], [4., 5., 6.]),
        ("cross product 3:", [4., 2., -3.], [-2., -5., 16.])
    ]

    logNDCross(name, u, v) = logResult(name, Algebra42.cross_product(u, v), log)

    benchNDCross(u, v) = Algebra42.cross_product(u, v)

    for (name, a, b) in cross_tests
        u, v = to_ndarrays(a, b)
        logNDCross(name, u, v)
        bench(name, benchNDCross, args=(u, v), function_used="cross")
    end
end


# ──── matrix multiplication ───────────────────────────────────────────────────────────────────── #

function test_nd_mul(log, bench)

    # function test_mul(dtype::Type, log)

    #     A = NDMatrix{Float64}([1. 0.; 0. 1.])
    #     v = NDVector{Float64}([4., 2.])

    #     logResult("mul 1:", Algebra42.mul(A, v), log)

    #     A = NDMatrix{Float64}([2. 0.; 0. 2.])
    #     v = NDVector{Float64}([4., 2.])

    #     logResult("mul 2:", Algebra42.mul(A, v), log)

    #     A = NDMatrix{Float64}([2. -2.; -2. 2.])
    #     v = NDVector{Float64}([4., 2.])

    #     logResult("mul 3:", Algebra42.mul(A, v), log)

    #     A = NDMatrix{Float64}([1. 0.; 0. 1.])
    #     B = NDMatrix{Float64}([1. 0.; 0. 1.])


    #     logResult("mul 4:", Algebra42.mul(A, B), log)

    #     A = NDMatrix{Float64}([1. 0.; 0. 1.])
    #     B = NDMatrix{Float64}([2. 1.; 4. 2.])

    #     logResult("mul 5:", Algebra42.mul(A, B), log)


    #     A = NDMatrix{Float64}([3. -5.; 6. 8.])
    #     B = NDMatrix{Float64}([2. 1.; 4. 2.])

    #     logResult("mul 6:", Algebra42.mul(A, B), log)

    # end


    testsMult = (
        ("mul 1:", [1.0 0.0; 0.0 1.0], [4.0, 2.0]),
        ("mul 2:", [2.0 0.0; 0.0 2.0], [4.0, 2.0]),
        ("mul 3:", [2.0 -2.0; -2.0 2.0], [4.0, 2.0]),
        ("mul 4:", [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]),
        ("mul 5:", [1.0 0.0; 0.0 1.0], [2.0 1.0; 4.0 2.0]),
        ("mul 6:", [3.0 -5.0; 6.0 8.0], [2.0 1.0; 4.0 2.0])
    )

    logNDMul(name, A, b) = logResult(name, A * b, log)
    benchNDMul(A, b) = A * b

    for (name, x, y) in testsMult
        A, b = to_ndarrays(x, y)
        logNDMul(name, A, b)
        bench(name, benchNDMul, args=(A, b), function_used="mat_mul")
    end
end


# ──── trace ───────────────────────────────────────────────────────────────────────────────────── #

function test_nd_trace(log, bench)

    # function test_trace(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. 0.; 0. 1.])

    #     logResult("trace 1:", Algebra42.trace(A), log)

    #     A = NDMatrix{Float64}([2. -5. 0.; 4. 3. 7.; -2. 3. 4.])

    #     logResult("trace 2:", Algebra42.trace(A), log)


    #     A = NDMatrix{Float64}([-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])

    #     logResult("trace 3:", Algebra42.trace(A), log)

    # end

    trace_tests = [
        ("trace 1:", [1. 0.; 0. 1.]),
        ("trace 2:", [2. -5. 0.; 4. 3. 7.; -2. 3. 4.]),
        ("trace 3:", [-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])
    ]

    bench_nd_trace(A) = Algebra42.trace(A)

    logNDTrace(name, A) = logResult(name, Algebra42.trace(A), log)

    for (name, a) in trace_tests
        A = to_ndarray(a)
        logNDTrace(name, A)
        bench(name, bench_nd_trace, args=(A,), function_used="trace")
    end


end




# ──── transpose ───────────────────────────────────────────────────────────────────────────────── #

function test_nd_transpose(log, bench)


    #     function test_transpose(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. 2.; 3. 4.])

    #     logResult("transpose 1:", Algebra42.transpose(A), log)

    #     A = NDMatrix{Float64}([2. -5. 0.; 4. 3. 7.; -2. 3. 4.])

    #     logResult("transpose 2:", Algebra42.transpose(A), log)


    #     A = NDMatrix{Float64}([-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])

    #     logResult("transpose 3:", Algebra42.transpose(A), log)

    #     A = NDMatrix{Float64}([1. 2.; 3. 4.; 5. 6.])

    #     logResult("transpose 4:", Algebra42.transpose(A), log)
    # end

    transpose_cases = [
        ("transpose 1:", [1.0 2.0; 3.0 4.0]),
        ("transpose 2:", [2.0 -5.0 0.0; 4.0 3.0 7.0; -2.0 3.0 4.0]),
        ("transpose 3:", [-2.0 -8.0 4.0; 1.0 -23.0 4.0; 0.0 6.0 4.0]),
        ("transpose 4:", [1.0 2.0; 3.0 4.0; 5.0 6.0])
    ]

    logNDTranspose(name, A) = logResult(name, Algebra42.transpose(A), log)

    benchNDTranspose(A) = Algebra42.transpose(A)

    for (name, a) in transpose_cases
        A = to_ndarray(a)
        logNDTranspose(name, A)
        bench(name, benchNDTranspose, args=(A,), function_used="transpose")
    end
end


# ──── reduced row echelon form ────────────────────────────────────────────────────────────────── #

function test_nd_row_echelon(log, bench)

    # function test_row_echelon(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    #     println("start row echelon1 : \n\n")

    #     println("A is $A")

    #     logResult("row_echelon 1:", Algebra42.reduced_row_echelon_form(A), log)

    #     A = NDMatrix{Float64}([1. 2.; 3. 4.])
    #     println("start row echelon2 : \n\n")
    #     logResult("row_echelon 2:", Algebra42.reduced_row_echelon_form(A), log)

    #     A = NDMatrix{Float64}([1. 2.; 2. 4.])
    #     println("start row echelon3 : \n\n")
    #     logResult("row_echelon 3:", Algebra42.reduced_row_echelon_form(A), log)

    #     println("start row echelon4: \n\n")
    #     A = NDMatrix{Float64}([8. 5. -2. 4. 28.; 4. 2.5 20. 4. -4.; 8. 5. 1. 4. 17.])
    #     logResult("row_echelon 4:", Algebra42.reduced_row_echelon_form(A), log)
    # end

    row_echelon_tests = [
        ("row_echelon 1:", [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]),
        ("row_echelon 2:", [1. 2.; 3. 4.]),
        ("row_echelon 3:", [1. 2.; 2. 4.]),
        ("row_echelon 4:", [8. 5. -2. 4. 28.; 4. 2.5 20. 4. -4.; 8. 5. 1. 4. 17.])
    ]

    logNDRowEch(name, A) = logResult(name, Algebra42.reduced_row_echelon_form(A), log)

    bench_nd_rref(A) = Algebra42.reduced_row_echelon_form(A)

    for (name, a) in row_echelon_tests
        A = to_ndarray(a)
        logNDRowEch(name, A)
        bench(name, bench_nd_rref, args=(A,), function_used="rref")
    end
end

# ──── determinant ─────────────────────────────────────────────────────────────────────────────── #


function test_nd_determinant(log, bench)

    # function test_determinant(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. -1.; -1. 1.])

    #     logResult("determinant 1:", Algebra42.determinant(A), log)

    #     A = NDMatrix{Float64}([2. 0. 0.; 0. 2. 0.; 0. 0. 2.])
    #     logResult("determinant 2:", Algebra42.determinant(A), log)

    #     A = NDMatrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.])
    #     logResult("determinant 3:", Algebra42.determinant(A), log)


    #     A = NDMatrix{Float64}([8. 5. -2. 4.; 4. 2.5 20. 4.; 8. 5. 1. 4.; 28. -4. 17. 1.])
    #     logResult("determinant 4:", Algebra42.determinant(A), log)
    # end

    # Define test matrices
    determinant_cases = [
        ("determinant 1:", [1.0 -1.0; -1.0 1.0]),
        ("determinant 2:", [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]),
        ("determinant 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0]),
        ("determinant 4:", [8.0 5.0 -2.0 4.0; 4.0 2.5 20.0 4.0; 8.0 5.0 1.0 4.0; 28.0 -4.0 17.0 1.0]
        )
    ]

    logNDDet(name, A) = logResult(name, Algebra42.determinant(A), log)
    benchNDDet(A) = Algebra42.determinant(A)

    for (name, a) in determinant_cases
        A = to_ndarray(a)
        logNDDet(name, A)
        bench(name, benchNDDet, args=(A,), function_used="det")
    end
end

# ──── inverse ─────────────────────────────────────────────────────────────────────────────────── #

function test_nd_inverse(log, bench)

    # function test_inverse(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    #     logResult("inverse 1:", Algebra42.inverse(A), log)

    #     A = NDMatrix{Float64}([2. 0. 0.; 0. 2. 0.; 0. 0. 2.])
    #     logResult("inverse 2:", Algebra42.inverse(A), log)

    #     A = NDMatrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.])
    #     logResult("inverse 3:", Algebra42.inverse(A), log)
    # end

    inverse_cases = [
        ("inverse 1:", [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
        ("inverse 2:", [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]),
        ("inverse 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0])
    ]

    logNDInverse(name, A) = logResult(name, Algebra42.inverse(A), log)

    benchNDInverse(A) = Algebra42.inverse(A)

    for (name, a) in inverse_cases
        A = to_ndarray(a)
        logNDInverse(name, A)
        bench(name, benchNDInverse, args=(A,), function_used="inverse")
    end
end

# ──── rank ────────────────────────────────────────────────────────────────────────────────────── #

function test_nd_rank(log, bench)

    # function test_rank(dtype::Type, log)
    #     A = NDMatrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    #     logResult("rank 1:", Algebra42.rank(A), log)

    #     A = NDMatrix{Float64}([1. 2. 0. 0.; 2. 4. 0. 0.; -1. 2. 1. 1.])
    #     logResult("rank 2:", Algebra42.rank(A), log)

    #     A = NDMatrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.; 21. 18. 7.])
    #     logResult("rank 3:", Algebra42.rank(A), log)
    # end

    rank_cases = [
        ("rank 1:", [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
        ("rank 2:", [1.0 2.0 0.0 0.0; 2.0 4.0 0.0 0.0; -1.0 2.0 1.0 1.0]),
        ("rank 3:", [8.0 5.0 -2.0; 4.0 7.0 20.0; 7.0 6.0 1.0; 21.0 18.0 7.0])
    ]

    logNDRank(name, A) = logResult(name, Algebra42.rank(A), log)

    benchNDRank(A) = Algebra42.rank(A)

    for (name, a) in rank_cases
        A = to_ndarray(a)
        logNDRank(name, A)
        bench(name, benchNDRank, args=(A,), function_used="rank")
    end
end



#println("=== Test 1: scalar with .+ ===")
function broadcast_test1()
    x = 2
    y = 3
    result = 0
    @MyBroadcast result = x .+ y
    @assert result == 5
end

function broadcast_test2()
    a = [1, 2, 3]
    b = [10, 20, 30]
    expected = [11, 22, 33]

    @MyBroadcast result = a .+ b
    @assert expected == result # [12,23,34]
end

function broadcast_test3()
    a = [1, 2, 3]
    b = [10, 20, 30]
    expected = [10, 40, 90]
    @MyBroadcast result = a .* b
    @assert result == expected   # [1*10, 2*20, 3*30] = [10,40, 90]
end


function broadcast_test4()
    a = [1, 2, 3]
    b = [10, 20, 30]
    result = Array{AbstractVector{Int},1}()
    expected = [[11, 22, 33], [11, 22, 33], [11, 22, 33]]
    @MyBroadcast begin
        for i in 1:3
            push!(result, a .+ b)
        end
    end

    @assert result == expected
end


function broadcast_test5()
    result = [0 0 0; 0 0 0; 0 0 0]
    a = [1 1 1; 2 2 2; 3 3 3]
    b = [10 10 10; 20 20 20; 30 30 30]

    expected = [11 11 11; 22 22 22; 33 33 33]

    @MyBroadcast begin
        result += a .+ b
    end

    @assert expected == result
end

function broadcast_test6()
    result = [0 0 0; 0 0 0; 0 0 0]
    a = [1 1 1; 2 2 2; 3 3 3]
    b = [10 10 10; 20 20 20; 30 30 30]
    expected = [11 11 11; 22 22 22; 33 33 33]

    @MyBroadcast begin
        result .+= a .+ b
    end

    @assert expected == result
end

function broadcast_test7()
    x = 10
    y = 4
    @MyBroadcast result = x .- y

    @assert 6 == result
end

function broadcast_test8()
    x = 20
    y = 5
    @MyBroadcast result = x ./ y
    @assert result == 4.0
end

function broadcast_test9()
    a = [1, 2, 3]
    b = [3, 4, 5]
    @MyBroadcast result = (a .+ b) .* (b .- a)
    @assert result == [8, 12, 16]
end

function broadcast_test10()
    a = [1, 2, 3]

    @MyBroadcast result = a .+ 10
    @assert result == [11, 12, 13]
end

function broadcast_test11()
    a = [1, 2, 3]

    @MyBroadcast result = 10 .+ a
    @assert result == [11, 12, 13]
end

function broadcast_test12()
    a = [1, 2, 3]
    b = [10, 10, 10]

    result = @MyBroadcast a .+ b
    @assert result == [11, 12, 13]
end

function broadcast_tests()
    broadcast_test1()
    broadcast_test2()
    broadcast_test3()
    broadcast_test4()
    broadcast_test5()
    broadcast_test6()
    broadcast_test7()
    broadcast_test8()
    broadcast_test9()
    broadcast_test10()
    broadcast_test11()
end

function run_tests_my_subject(log, bench)
    test_nd_add_function(log, bench)
    test_nd_sub_function(log, bench)
    test_nd_prod_function(log, bench) # scalar
    test_nd_linear_combination(log, bench)
    test_nd_lerp(log, bench)
    test_nd_dot(log, bench)
    test_nd_norm(log, bench)
    test_nd_angle_cos(log, bench)
    test_nd_cross_product(log, bench)
    test_nd_mul(log, bench)
    test_nd_trace(log, bench)
    test_nd_transpose(log, bench)
    test_nd_row_echelon(log, bench)
    test_nd_determinant(log, bench)
    test_nd_inverse(log, bench)
    test_nd_rank(log, bench)
    broadcast_tests()
end


