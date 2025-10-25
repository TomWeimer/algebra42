using Algebra42
using .TestData
using .TestUtils

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (shape, content)                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


function logNDArrayContent(name, ndarray, write_log_fn)
    write_log_fn("\n" * name)
    write_content(ndarray, write_log_fn)
end

function test1(io::IO, v, opn='[', cls=']')
    prefix, implicit = Base.typeinfo_prefix(io, v)
    print(io, prefix)
    # directly or indirectly, the context now knows about eltype(v)
    if !implicit
        io = Base.IOContext(io, :typeinfo => eltype(v))
    end
    limited = Base.get(io, :limit, false)::Bool

    if limited && length(v) > 20
        axs1 = Base.axes1(v)
        f, l = first(axs1), last(axs1)
        test2(io, v, opn, ",", "", false, f, f+9)
        print(io, "  …  ")
        test2(io, v, "", ",", cls, false, l-9, l)
    else
        test2(io, v, opn, ",", cls, false)
    end
end


function test2(io::IO, itr::Union{AbstractArray}, op, delim, cl,
                          delim_one, i1=first(LinearIndices(itr)), l=last(LinearIndices(itr)))
    print(io, op)
    if !Base.show_circular(io, itr)
        recur_io = IOContext(io, :SHOWN_SET => itr)
        first = true
        i = i1 
        a =string(Base.first(LinearIndices(itr)))
        b =string(Base.last(LinearIndices(itr)))
        d = last(Base.axes1(itr.indices[1]))
        c =  Base.first(Base.axes1(itr.indices[1]))

        if l >= i1
            while true
                if !isassigned(itr, i)
                    print(io, undef_ref_str)
                else
                    x = itr[i]
                    show(recur_io, x)
                end
                if i == l
                    delim_one && first && print(io, delim)
                    break
                end
                i += 1
                first = false
                print(io, delim)
                print(io, ' ')
            end
        end
    end
    print(io, cl)
end

# empty ndarray:

function testEmptyNDArray(name, test::NDArrayData, write_log_fn)
    array = NDArray{Int}(test.array)

    function myall(f, itr)
         anymissing = false
         
        for x in itr
            v = f(x)
            if ismissing(v)
                anymissing = true
            else
                v || return false
            end
        end
        return anymissing ? missing : true
    end

    A = array[:]
    test1(stdout, A)


    logNDArrayContent(name, array, write_log_fn)

    exceptions = [
        (() -> array[1], BoundsError),
        (() -> array[2], BoundsError),
        (() -> array[1, :], BoundsError)
    ]

    # Check iteration
    nothingHappen = myall(x -> false, array)

    @assert allExceptionPassed(exceptions) && nothingHappen
end

# scalar

function testScalarNDArray(name, test::TestDataScalar, write_log_fn)
    array = NDArray{Int}(test.data)

    logNDArrayContent(name, array, write_log_fn)
    a = array[:]
    @assert exceptionPassed(() -> array[2], 1, DomainError)

end


# 'ragged' ndarray:

function testRaggedArray(name, test::NDArrayData, write_log_fn)

    array = NDArray{Any}(test.array)

    logNDArrayContent(name, array, write_log_fn)

    f1 = () -> x = NDArray{Int}(test.array)
    errorThrown = exceptionPassed(f1, 1, DomainError)

    @assert errorThrown
end


function ndarray_test_set1(write_log_fn)
    testScalarNDArray("scalar", scalar, write_log_fn)
    logNDArrayContent("vector", vector_1, write_log_fn)
    testEmptyNDArray("empty", empty_vector, write_log_fn)
    logNDArrayContent("matrix 1", matrix_1, write_log_fn)
    logNDArrayContent("matrix 2", matrix_2,  write_log_fn)
    logNDArrayContent("row", row_matrix,  write_log_fn)
    logNDArrayContent("column", column_matrix,  write_log_fn)
    logNDArrayContent("array3D 1", array_3D_1,  write_log_fn)
    logNDArrayContent("array3D 2", array_3D_2,  write_log_fn)
    testRaggedArray("ragged array", ragged_array,  write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests NDArray (formats numpy and julia)                                         #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function testBothFormat(data::NDArrayData)
    format1, format2 = NDArray{Int}(data.array), NDArray{Int}(data.matrix)
    passed = size(format1) == size(format2) && compareByIndex(format1, format2)
    return passed
end

function ndarray_test_set2(write_log_fn)
    @assert testBothFormat(vector_1)
    @assert testBothFormat(empty_vector)
    @assert testBothFormat(matrix_1)
    @assert testBothFormat(matrix_2)
    @assert testBothFormat(row_matrix)
    @assert testBothFormat(column_matrix)
    @assert testBothFormat(array_3D_1)
    @assert testBothFormat(array_3D_2)
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (slices)                                                     #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


# 'scalar' ndarray
function testScalarSlices(name, test::TestDataScalar, indices, write_log_fn)
    ndarray = NDArray{Int}(test.data)

    logNDArrayContent(name, ndarray, write_log_fn)
end

# 'normal' ndarray

function testNDArraySlices(name, ndarray::NDArray, ndarrayToChange::NDArray, indices, write_log_fn)
    sliced, slicedToChange = ndarray[indices...], ndarrayToChange[indices...]

    logNDArrayContent(name, sliced, write_log_fn)

    if (isempty(sliced))
        return
    end


    idx1, idx2 = ntuple(_ -> 1, ndims(ndarray)), ntuple(_ -> 1, ndims(sliced))
    mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
    passByRef = (ndarrayToChange[idx1...] == 10)


    mutableOk || println("mutable: ", mutableOk, " should be ", true)
    passByRef || println("passByRef: ", passByRef, " should be ", true)

    @assert mutableOk && passByRef
end

function testNDArraySlices(name, test::NDArrayData, indices, write_log_fn)
    ndarray = NDArray{Int}(test.array)
    ndarrayToChange = NDArray{Int}(test.array)
    return testNDArraySlices(name, ndarray, ndarrayToChange, indices, write_log_fn)
end

function testRaggedArraySlices(name, test::NDArrayData, indices, write_log_fn)
    ndarray = NDArray{Any}(test.array)
    ndarrayToChange = NDArray{Any}(test.array)
    return testNDArraySlices(name, ndarray, ndarrayToChange, indices, write_log_fn)
end


function mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, val)
    slicedToChange[idx2...] = val
    return ndarrayToChange[idx1...] == val
end


function ndarray_test_set3(write_log_fn)
    testScalarSlices("scalar", scalar, (:,), write_log_fn)
    testNDArraySlices("vector", vector_1, (:,), write_log_fn)
    testNDArraySlices("matrix", matrix_1, (1, :), write_log_fn)
    testNDArraySlices("array3D", array_3D_1, (1, 1, :), write_log_fn)
    testNDArraySlices("emptyArray", empty_vector, (:,), write_log_fn)
    testNDArraySlices("columnVector", column_matrix, (1, :), write_log_fn)
    testNDArraySlices("rowVector", row_matrix, (1, :), write_log_fn)
    testRaggedArraySlices("raggedArray", ragged_array, (:,), write_log_fn)
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                       Tests NDArray (all sets)                                                   #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function run_tests_my_ndarray(write_log_fn)
    ndarray_test_set1(write_log_fn)
    ndarray_test_set2(write_log_fn)
    ndarray_test_set3(write_log_fn)
end