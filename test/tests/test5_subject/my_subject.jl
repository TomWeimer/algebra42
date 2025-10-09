using Algebra42
using .TestData
using .TestUtils

using Debugger

function logResult(name, result::String, write_log_fn)
    write_log_fn("\n" * name)
    write_log_fn(result)
end

function logResult(name, result, write_log_fn)
    write_log_fn("\n" * name)
    write_content(result, write_log_fn)
end

function logAddResult(name, elem1, elem2, write_log_fn)
    logResult(name, Algebra42.add(elem1, elem2), write_log_fn) 
end

function logSubResult(name, elem1, elem2, write_log_fn)
    logResult(name, Algebra42.sub(elem1, elem2), write_log_fn) 
end

function logProdResult(name, elem1, scalar::Number, write_log_fn)
    logResult(name, Algebra42.prod(elem1, scalar), write_log_fn) 
end

function mustThrowAdd(elem1, elem2)
    f = () -> Algebra42.add(elem1, elem2)

    @assert exceptionPassed(f, 1, DomainError)     
end

function mustThrowSub(elem1, elem2)
    f = () -> Algebra42.sub(elem1, elem2)

    @assert exceptionPassed(f, 1, DomainError)     
end

function mustThrowProd(elem1, scalar)
    f = () -> Algebra42.prod(scalar, elem1)

    @assert exceptionPassed(f, 1, MethodError)     
end

function test_add_function(dtype::Type, write_log_fn)
    logAddResult("vector + vector 1", Algebra42.Vector{dtype}(vector_3.array),  Algebra42.Vector{dtype}(vector_4.array), write_log_fn)
    logAddResult("vector + vector 2", Algebra42.Vector{dtype}(vector_3.array),  Algebra42.Vector{dtype}(vector_3.array), write_log_fn)
    mustThrowAdd(Algebra42.Vector{dtype}(vector_4.array), Algebra42.Vector{dtype}(vector_5.array))
    mustThrowAdd(Algebra42.Vector{dtype}(vector_3.array), Algebra42.Vector{dtype}(vector_5.array))
    mustThrowAdd(Algebra42.Vector{dtype}(vector_1.array), Algebra42.Vector{dtype}(vector_4.array))
    mustThrowAdd(Algebra42.Vector{dtype}(vector_4.array), Algebra42.Vector{dtype}(vector_1.array))
end

function test_sub_function(dtype::Type, write_log_fn)
    logSubResult("vector - vector 1", Algebra42.Vector{dtype}(vector_3.array),  Algebra42.Vector{dtype}(vector_4.array), write_log_fn)
    logSubResult("vector - vector 2", Algebra42.Vector{dtype}(vector_3.array),  Algebra42.Vector{dtype}(vector_3.array), write_log_fn)
    mustThrowSub(Algebra42.Vector{dtype}(vector_4.array), Algebra42.Vector{dtype}(vector_5.array))
    mustThrowSub(Algebra42.Vector{dtype}(vector_3.array), Algebra42.Vector{dtype}(vector_5.array))
    mustThrowSub(Algebra42.Vector{dtype}(vector_1.array), Algebra42.Vector{dtype}(vector_4.array))
    mustThrowSub(Algebra42.Vector{dtype}(vector_4.array), Algebra42.Vector{dtype}(vector_1.array))
end

# scalar
function test_prod_function(dtype::Type, write_log_fn)
    logProdResult("vector * 42 1", Algebra42.Vector{dtype}(vector_3.array), scalar.data, write_log_fn)
    logProdResult("vector * 42 2", Algebra42.Vector{dtype}(vector_4.array), scalar.data, write_log_fn)
    mustThrowProd(Algebra42.Vector{dtype}(vector_4.array), scalar.data)
end

# scalar
function test_linear_combination(dtype::Type, write_log_fn)

    e1 = Algebra42.Vector{Float64}([1., 0., 0.])
    e2 = Algebra42.Vector{Float64}([0., 1., 0.])
    e3 = Algebra42.Vector{Float64}([0., 0., 1.])
    coefs = [10., -2., 0.5]
    logResult("linear comb 1:", Algebra42.linear_combination([e1, e2, e3], coefs), write_log_fn)

    v1 = Algebra42.Vector{Float64}([1., 2., 3.])
    v2 = Algebra42.Vector{Float64}([0., 10., -100.])
    coefs = [10., -2.]
    logResult("linear comb 2:", Algebra42.linear_combination([v1, v2], coefs), write_log_fn)
end

# scalar
function test_lerp(dtype::Type, write_log_fn)
    logResult("lerp 1:", Algebra42.lerp(0., 1., 0.), write_log_fn)
    logResult("lerp 2:", Algebra42.lerp(0., 1., 1.), write_log_fn)
    logResult("lerp 3:", Algebra42.lerp(0., 1., 0.5), write_log_fn)
    logResult("lerp 4:", Algebra42.lerp(21., 42., 0.3), write_log_fn)


    u = Algebra42.Vector{Float64}([2., 1.])
    v = Algebra42.Vector{Float64}([4., 2.])

    logResult("lerp 5:",  Algebra42.lerp(u, v, 0.3), write_log_fn)

    A = Algebra42.Matrix{Float64}([2. 1.; 3. 4.])
    B = Algebra42.Matrix{Float64}([20. 10.; 30. 40.])
    
    logResult("lerp 6:",  Algebra42.lerp(A, B, 0.5), write_log_fn)
end

function test_dot(dtype::Type, write_log_fn)

    u = Algebra42.Vector{Float64}([0., 0.])
    v = Algebra42.Vector{Float64}([1., 1.])

    logResult("dot 1:",  Algebra42.dot(u, v), write_log_fn)

    u = Algebra42.Vector{Float64}([1., 1.])
    v = Algebra42.Vector{Float64}([1., 1.])

    logResult("dot 2:",  Algebra42.dot(u, v), write_log_fn)

    u = Algebra42.Vector{Float64}([-1., 6.])
    v = Algebra42.Vector{Float64}([3., 2.])

    logResult("dot 3:",  Algebra42.dot(u, v), write_log_fn)
end

function test_norm(dtype::Type, write_log_fn)
    u = Algebra42.Vector{Float64}([0., 0., 0.])

    logResult("norms 1:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", write_log_fn);

    u = Algebra42.Vector{Float64}([1., 2., 3.])

    logResult("norms 2:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", write_log_fn);

    u = Algebra42.Vector{Float64}([-1., -2.])

    logResult("norms 3:", "$(norm_1(u)), $(Algebra42.norm(u)), $(norm_inf(u))", write_log_fn);

end

function test_angle_cos(dtype::Type, write_log_fn)
    u = Algebra42.Vector{Float64}([1., 0.])
    v = Algebra42.Vector{Float64}([1., 0.])

    logResult("angle cos 1:", Algebra42.angle_cos(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([1., 0.])
    v = Algebra42.Vector{Float64}([0., 1.])

    logResult("angle cos 2:", Algebra42.angle_cos(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([-1., 1.])
    v = Algebra42.Vector{Float64}([1., -1.])

    logResult("angle cos 3:", Algebra42.angle_cos(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([2., 1.])
    v = Algebra42.Vector{Float64}([4., 2.])

    logResult("angle cos 4:", Algebra42.angle_cos(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([1., 2., 3.])
    v = Algebra42.Vector{Float64}([4., 5., 6.])

    logResult("angle cos 5:", Algebra42.angle_cos(u, v), write_log_fn);
end

function test_cross_product(dtype::Type, write_log_fn)
    u = Algebra42.Vector{Float64}([0., 0., 1.])
    v = Algebra42.Vector{Float64}([1., 0., 0.])

    logResult("cross product 1:", Algebra42.cross_product(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([1., 2., 3.])
    v = Algebra42.Vector{Float64}([4., 5., 6.])

    logResult("cross product 2:", Algebra42.cross_product(u, v), write_log_fn);

    u = Algebra42.Vector{Float64}([4., 2., -3.])
    v = Algebra42.Vector{Float64}([-2., -5., 16.])

    logResult("cross product 3:", Algebra42.cross_product(u, v), write_log_fn);
  
    
end

function test_mul(dtype::Type, write_log_fn)

    A = Algebra42.Matrix{Float64}([1. 0.; 0. 1.])
    v = Algebra42.Vector{Float64}([4., 2.])

    logResult("mul 1:",  Algebra42.mul(A, v), write_log_fn)

    A = Algebra42.Matrix{Float64}([2. 0.; 0. 2.])
    v = Algebra42.Vector{Float64}([4., 2.])

    logResult("mul 2:",  Algebra42.mul(A, v), write_log_fn)

    A = Algebra42.Matrix{Float64}([2. -2.; -2. 2.])
    v = Algebra42.Vector{Float64}([4., 2.])

    logResult("mul 3:",  Algebra42.mul(A, v), write_log_fn)

    A = Algebra42.Matrix{Float64}([1. 0.; 0. 1.])
    B = Algebra42.Matrix{Float64}([1. 0.; 0. 1.])


    logResult("mul 4:",  Algebra42.mul(A, B), write_log_fn)

    A = Algebra42.Matrix{Float64}([1. 0.; 0. 1.])
    B = Algebra42.Matrix{Float64}([2. 1.; 4. 2.])

    logResult("mul 5:",  Algebra42.mul(A, B), write_log_fn)


    A = Algebra42.Matrix{Float64}([3. -5.; 6. 8.])
    B = Algebra42.Matrix{Float64}([2. 1.; 4. 2.])

    logResult("mul 6:",  Algebra42.mul(A, B), write_log_fn)

end

function test_trace(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. 0.; 0. 1.])

    logResult("trace 1:", Algebra42.trace(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([2. -5. 0.; 4. 3. 7.; -2. 3. 4.])

    logResult("trace 2:", Algebra42.trace(A), write_log_fn);


    A = Algebra42.Matrix{Float64}([-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])

    logResult("trace 3:", Algebra42.trace(A), write_log_fn);

end


function test_transpose(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. 2.; 3. 4.])

    logResult("transpose 1:", Algebra42.transpose(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([2. -5. 0.; 4. 3. 7.; -2. 3. 4.])

    logResult("transpose 2:", Algebra42.transpose(A), write_log_fn);


    A = Algebra42.Matrix{Float64}([-2. -8. 4.; 1. -23. 4.; 0. 6. 4.])

    logResult("transpose 3:", Algebra42.transpose(A), write_log_fn);
end


function test_row_echelon(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    logResult("row_echelon 1:", Algebra42.reduced_row_echelon_form(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([1. 2.; 3. 4.])
    logResult("row_echelon 2:", Algebra42.reduced_row_echelon_form(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([1. 2.; 2. 4.])
    logResult("row_echelon 3:", Algebra42.reduced_row_echelon_form(A), write_log_fn);


    A = Algebra42.Matrix{Float64}([8. 5. -2. 4. 28.; 4. 2.5 20. 4. -4. ; 8. 5. 1. 4. 17.])
    logResult("row_echelon 4:", Algebra42.reduced_row_echelon_form(A), write_log_fn);
end

function test_determinant(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. -1.; -1. 1.])

    logResult("determinant 1:", Algebra42.determinant(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([2. 0. 0.; 0. 2. 0.; 0. 0. 2.])
    logResult("determinant 2:", Algebra42.determinant(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.])
    logResult("determinant 3:", Algebra42.determinant(A), write_log_fn);


    A = Algebra42.Matrix{Float64}([8. 5. -2. 4.; 4. 2.5 20. 4.; 8. 5. 1. 4.; 28. -4. 17. 1. ])
    logResult("determinant 4:", Algebra42.determinant(A), write_log_fn);
end

function test_inverse(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    logResult("inverse 1:", Algebra42.inverse(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([2. 0. 0.; 0. 2. 0.; 0. 0. 2.])
    logResult("inverse 2:", Algebra42.inverse(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.])
    logResult("inverse 3:", Algebra42.inverse(A), write_log_fn);
end


function test_rank(dtype::Type, write_log_fn)
    A = Algebra42.Matrix{Float64}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    logResult("rank 1:", Algebra42.rank(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([1. 2. 0. 0.; 2. 4. 0. 0.; -1. 2. 1. 1.])
    logResult("rank 2:", Algebra42.rank(A), write_log_fn);

    A = Algebra42.Matrix{Float64}([8. 5. -2.; 4. 7. 20.; 7. 6. 1.; 21. 18. 7.])
    logResult("rank 3:", Algebra42.rank(A), write_log_fn);
end

function run_tests_my_subject(write_log_fn)
    test_add_function(Int, write_log_fn)
    test_sub_function(Int, write_log_fn)
    test_prod_function(Int, write_log_fn) # scalar
    test_linear_combination(Int, write_log_fn)
    test_lerp(Int, write_log_fn)
    test_dot(Int, write_log_fn)
    test_norm(Int, write_log_fn)
    test_angle_cos(Int, write_log_fn)
    test_cross_product(Int, write_log_fn)
    test_mul(Int, write_log_fn)
    test_trace(Int, write_log_fn)
    test_transpose(Int, write_log_fn)
    test_row_echelon(Int, write_log_fn)
    #test_determinant(Int, write_log_fn)
    test_inverse(Int, write_log_fn)
    test_rank(Int, write_log_fn)
end
