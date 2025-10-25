using .TestData
using .TestUtils


using Pkg

Pkg.add("GeometryBasics")

Pkg.add("LinearAlgebra")

Pkg.add("RowEchelon")

using GeometryBasics
using LinearAlgebra
using RowEchelon
using Debugger

function logResult(name, result::String, write_log_fn)
    write_log_fn("\n" * name)
    write_log_fn(result)
end

function logResult(name, result, write_log_fn)
    write_log_fn("\n" * name)
    write_content(result, write_log_fn)
end

function logAddResult(name, elem1::Base.Matrix, elem2::Base.Matrix, write_log_fn)
    logResult(name, elem1 .+ elem2, write_log_fn) 
end

function logSubResult(name, elem1::Base.Matrix, elem2::Base.Matrix, write_log_fn)
    logResult(name, elem1 .- elem2, write_log_fn) 
end

function logProdResult(name, elem1::Base.Matrix, scalar::Number, write_log_fn)
    logResult(name, elem1 .* scalar, write_log_fn) 
end

function logAddResult(name, elem1::Base.Vector, elem2::Base.Vector, write_log_fn)
    logResult(name, elem1 .+ elem2, write_log_fn) 
end

function logSubResult(name, elem1::Base.Vector, elem2::Base.Vector, write_log_fn)
    logResult(name, elem1 .- elem2, write_log_fn) 
end

function logProdResult(name, elem1::Base.Vector, scalar::Number, write_log_fn)
    logResult(name, elem1 .* scalar, write_log_fn) 
end


function test_add_function( write_log_fn)
    logAddResult("vector + vector 1", vector_3.matrix,  vector_4.matrix, write_log_fn)
    logAddResult("vector + vector 2", vector_3.matrix,  vector_3.matrix, write_log_fn)
end

function test_sub_function(write_log_fn)
    logSubResult("vector - vector 1", vector_3.matrix,  vector_4.matrix, write_log_fn)
    logSubResult("vector - vector 2", vector_3.matrix,  vector_3.matrix, write_log_fn)
end

# scalar
function test_prod_function( write_log_fn)
    logProdResult("vector * 42 1", vector_3.matrix, scalar.data, write_log_fn)
    logProdResult("vector * 42 2", vector_4.matrix, scalar.data, write_log_fn)
end

# scalar
function test_linear_combination( write_log_fn)

    e1 = [1., 0., 0.]
    e2 = [0., 1., 0.]
    e3 = [0., 0., 1.]

    vectors = [e1, e2, e3]
    coefs = [10., -2., 0.5]
    logResult("linear comb 1:", sum(a .* v for (a, v) in zip(coefs, vectors)), write_log_fn)

    v1 = [1., 2., 3.]
    v2 = [0., 10., -100.]
    vectors = [v1, v2]
    coefs = [10., -2.]
    logResult("linear comb 2:", sum(a .* v for (a, v) in zip(coefs, vectors)), write_log_fn)
end

# scalar
function test_lerp(write_log_fn)
    # 1D scalar interpolation
    logResult("lerp 1:", 0.0, write_log_fn)
    logResult("lerp 2:", 1.0, write_log_fn)
    logResult("lerp 3:", 0.5, write_log_fn)  # position in domain
    logResult("lerp 4:", 27.3, write_log_fn)
    logResult("lerp 5:", [2.6, 1.3], write_log_fn)
    logResult("lerp 6:", [11. 5.5; 16.5 22.], write_log_fn)
end

function test_dot( write_log_fn)

    u = [0., 0.]
    v = [1., 1.]

    logResult("dot 1:",  LinearAlgebra.dot(u, v), write_log_fn)

    u = [1., 1.]
    v = [1., 1.]

    logResult("dot 2:",  LinearAlgebra.dot(u, v), write_log_fn)

    u = [-1., 6.]
    v = [3., 2.]

    logResult("dot 3:",  LinearAlgebra.dot(u, v), write_log_fn)
end

function test_norm(write_log_fn)
    u = [0., 0., 0.]

    logResult("norms 1:", "$(LinearAlgebra.norm(u, 1)), $(LinearAlgebra.norm(u)), $(LinearAlgebra.norm(u, Inf))", write_log_fn);

    u = [1., 2., 3.]

    logResult("norms 2:", "$(LinearAlgebra.norm(u, 1)), $(LinearAlgebra.norm(u)), $(LinearAlgebra.norm(u, Inf))", write_log_fn);

    u = [-1., -2.]

    logResult("norms 3:", "$(LinearAlgebra.norm(u, 1)), $(LinearAlgebra.norm(u)), $(LinearAlgebra.norm(u, Inf))", write_log_fn);

end

function test_angle_cos( write_log_fn)

    angle = (a, b) -> LinearAlgebra.dot(a,b)/( LinearAlgebra.norm(a)* LinearAlgebra.norm(b))

    u = [1., 0.]
    v = [1., 0.]

    logResult("angle cos 1:", angle(u, v), write_log_fn);

    u = [1., 0.]
    v = [0., 1.]

    logResult("angle cos 2:", angle(u, v), write_log_fn);

    u = [-1., 1.]
    v = [1., -1.]

    logResult("angle cos 3:", angle(u, v), write_log_fn);

    u = [2., 1.]
    v = [4., 2.]

    logResult("angle cos 4:", angle(u, v), write_log_fn);

    u = [1., 2., 3.]
    v = [4., 5., 6.]

    logResult("angle cos 5:", angle(u, v), write_log_fn);
end

function test_cross_product( write_log_fn)
    u = [0., 0., 1.]
    v = [1., 0., 0.]

    logResult("cross product 1:", LinearAlgebra.cross(u, v), write_log_fn);

    u = [1., 2., 3.]
    v = [4., 5., 6.]

    logResult("cross product 2:", LinearAlgebra.cross(u, v), write_log_fn);

    u = [4., 2., -3.]
    v = [-2., -5., 16.]

    logResult("cross product 3:", LinearAlgebra.cross(u, v), write_log_fn);
  
    
end

function test_mul( write_log_fn)

    A = [1. 0.; 0. 1.]
    v = [4., 2.]

    logResult("mul 1:",  A * v, write_log_fn)

    A = [2. 0.; 0. 2.]
    v = [4., 2.]

    logResult("mul 2:", A * v, write_log_fn)

    A = [2. -2.; -2. 2.]
    v = [4., 2.]

    logResult("mul 3:", A * v, write_log_fn)

    A = [1. 0.; 0. 1.]
    B = [1. 0.; 0. 1.]


    logResult("mul 4:",  A * B, write_log_fn)

    A = [1. 0.; 0. 1.]
    B = [2. 1.; 4. 2.]

    logResult("mul 5:", A * B, write_log_fn)


    A = [3. -5.; 6. 8.]
    B = [2. 1.; 4. 2.]

    logResult("mul 6:",  A * B, write_log_fn)

end

function test_trace(write_log_fn)
    A = [1. 0.; 0. 1.]

    logResult("trace 1:", LinearAlgebra.tr(A), write_log_fn);

    A = [2. -5. 0.; 4. 3. 7.; -2. 3. 4.]

    logResult("trace 2:", LinearAlgebra.tr(A), write_log_fn);


    A = [-2. -8. 4.; 1. -23. 4.; 0. 6. 4.]

    logResult("trace 3:", LinearAlgebra.tr(A), write_log_fn);

end


function test_transpose( write_log_fn)
    A = [1. 2.; 3. 4.]

    logResult("transpose 1:", LinearAlgebra.transpose(A), write_log_fn);

    A = [2. -5. 0.; 4. 3. 7.; -2. 3. 4.]

    logResult("transpose 2:", LinearAlgebra.transpose(A), write_log_fn);


    A = [-2. -8. 4.; 1. -23. 4.; 0. 6. 4.]

    logResult("transpose 3:", LinearAlgebra.transpose(A), write_log_fn);

    A = [1. 2.; 3. 4.; 5. 6.]

    logResult("transpose 4:", transpose(A), write_log_fn);
end


function test_row_echelon( write_log_fn)
    A = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]

    logResult("row_echelon 1:", RowEchelon.rref(A), write_log_fn);

    A = [1. 2.; 3. 4.]
    logResult("row_echelon 2:", RowEchelon.rref(A), write_log_fn);

    A = [1. 2.; 2. 4.]
    logResult("row_echelon 3:", RowEchelon.rref(A), write_log_fn);


    A = [8. 5. -2. 4. 28.; 4. 2.5 20. 4. -4. ; 8. 5. 1. 4. 17.]
    logResult("row_echelon 4:", RowEchelon.rref(A), write_log_fn);
end

function test_determinant( write_log_fn)
    A = [1. -1.; -1. 1.]

    logResult("determinant 1:", LinearAlgebra.det(A), write_log_fn);

    A = [2. 0. 0.; 0. 2. 0.; 0. 0. 2.]
    logResult("determinant 2:", LinearAlgebra.det(A), write_log_fn);

    A = [8. 5. -2.; 4. 7. 20.; 7. 6. 1.]
    logResult("determinant 3:", LinearAlgebra.det(A), write_log_fn);


    A = [8. 5. -2. 4.; 4. 2.5 20. 4.; 8. 5. 1. 4.; 28. -4. 17. 1. ]
    logResult("determinant 4:", LinearAlgebra.det(A), write_log_fn);
end

function test_inverse( write_log_fn)
    A = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]

    logResult("inverse 1:", LinearAlgebra.inv(A), write_log_fn);

    A = [2. 0. 0.; 0. 2. 0.; 0. 0. 2.]
    logResult("inverse 2:", LinearAlgebra.inv(A), write_log_fn);

    A = [8. 5. -2.; 4. 7. 20.; 7. 6. 1.]
    logResult("inverse 3:", LinearAlgebra.inv(A), write_log_fn);
end


function test_rank(write_log_fn)
    A = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]

    logResult("rank 1:", LinearAlgebra.rank(A), write_log_fn);

    A = [1. 2. 0. 0.; 2. 4. 0. 0.; -1. 2. 1. 1.]
    logResult("rank 2:", LinearAlgebra.rank(A), write_log_fn);

    A = [8. 5. -2.; 4. 7. 20.; 7. 6. 1.; 21. 18. 7.]
    logResult("rank 3:", LinearAlgebra.rank(A), write_log_fn);
end

function run_tests_julia_subject(write_log_fn)
    test_add_function(write_log_fn)
    test_sub_function(write_log_fn)
    test_prod_function(write_log_fn) # scalar
    test_linear_combination(write_log_fn)
    test_lerp(write_log_fn)
    test_dot(write_log_fn)
    test_norm(write_log_fn)
    test_angle_cos(write_log_fn)
    test_cross_product(write_log_fn)
    test_mul(write_log_fn)
    test_trace(write_log_fn)
    test_transpose(write_log_fn)
    test_row_echelon(write_log_fn)
    test_determinant(write_log_fn)
    test_inverse(write_log_fn)
    test_rank(write_log_fn)
end