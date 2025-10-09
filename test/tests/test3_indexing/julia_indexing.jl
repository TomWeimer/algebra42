using Algebra42
using .TestData

function logIndexing(name, result, write_log_fn)
    write_log_fn("\n" * name)
    write_content(result, write_log_fn)
end

function test_shape_incompatibility(name, data::TestDataArray, write_log_fn, indices...)
    passed = false
    expanded_indices = []
    for index in indices
        push!(expanded_indices, (index isa AbstractArray{Bool}) ? findall(index) : index)
    end
    try
        tmp = data.matrix[expanded_indices...]
        logIndexing(name, tmp, write_log_fn)
    catch e
        passed = e isa DomainError
    end
    
    @assert passed
end

function test_fancy_indexing(name, data::TestDataArray, write_log_fn, indices...)
    expanded_indices = []
    for index in indices
        push!(expanded_indices, (index isa AbstractArray{Bool}) ? findall(index) : index)
    end
    got = data.matrix[expanded_indices...]
    logIndexing(name, got, write_log_fn)
end

function run_tests_julia_indexing(write_log_fn)
    # TODO
    ndarray1 = [1 1; 1 1]
    ndarray2 = [2 2; 2 2]

    ndarray3 = ndarray1 + ndarray2

    logIndexing("array: ", ndarray3, write_log_fn)
    #test_fancy_indexing("Int indexing", array_4D_1, write_log_fn, [1, 2], [1 2; 2 1])
    # test_fancy_indexing("Bool mask 1", vector_1, write_log_fn, [true, false, true, true])
    # test_fancy_indexing("Bool mask 2", matrix_1, write_log_fn, [true false true; true true false])
    # test_fancy_indexing("Bool mask 3", matrix_1, write_log_fn,  [false false false; false false false])
    # test_fancy_indexing("Bool mask 4", array_3D_1, write_log_fn,[false, true], [false, true], [false, true])
    # test_fancy_indexing("Bool mask 5", array_3D_1, write_log_fn, [false, false], [false, false], [false, false])
    # test_fancy_indexing("Bool indexing", array_4D_1, write_log_fn, [true false; false true], [false true; true false])
   # test_shape_incompatibility("Shape invalid", array_4D_1, write_log_fn, [1, 2, 3], [1 2; 2 1])
    #test_fancy_indexing("Mixed indexing 1", array_4D_1, write_log_fn, [true false; true false], [1 2; 1 2] )
   # test_shape_incompatibility("Mixed indexing 2", array_4D_1, write_log_fn, [false false; false false], [1 2; 2 1] )
end