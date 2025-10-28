module TestUtils

using Algebra42

using ..TestData

export printTest, exceptionPassed, verifyShape, compareArrayContent, compareByIndex, compareByIterator, compareBySlices, exceptionPassed, allExceptionPassed, comparesSlices, write_content, logByIndex, logByIterator, logBySlices


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests Utils ( printing tests )                                                  #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function printTest(name::AbstractString, result::Bool)
    try
        if result === true
            println("   ✅ Test passed: $name")
        else
            println("   ❌ Test failed: $name")
        end
    catch e
        println("   ❌ Test error: $name")
        println("      → $(typeof(e)): $(e.msg)")
    end
end

function test(name::AbstractString, result::Bool)
    try
        if result === true
            println("   ✅ Test passed: $name")
        else
            println("   ❌ Test failed: $name")
        end
    catch e
        println("   ❌ Test error: $name")
        println("      → $(typeof(e)): $(e.msg)")
    end
end


function pretty_index(idx...)
    parts = map(pretty_index, idx)
    return join(parts, ", ")
end

function pretty_index(idx::AbstractArray)
    parts = map(v -> 
        v === Colon() ? ":" : 
        string(v),
        idx
    )
    return "[" * join(parts, ", ") * "]"
end

function pretty_index(idx::Tuple)
    parts = map(v -> 
        v === Colon() ? ":" : 
        isa(v, AbstractArray) ? pretty_index(v) : 
        string(v),
        idx
    )
    return "(" * join(parts, ", ") * ")"
end

function pretty_index(idx::Colon)
    return ":"
end

function pretty_index(idx::Number)
    return string(idx)
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests Utils ( check shape )                                                     #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#


function verifyShape(shape::Shape, expected::ShapeData)
    if shape.dims != expected.shape
        println("Shape mismatch: got $(shape.dims), expected $(expected.shape)")
    end

    if shape.length != expected.total_element
        println("Size mismatch: got $(shape.length), expected $(expected.total_element)")
    end

    if ndims(shape) != expected.dim
        println("Dim mismatch: got $(ndims(shape)), expected $(expected.dim)")
    end

    return shape.dims == expected.shape &&
           shape.length == expected.total_element &&
           ndims(shape) == expected.dim
end

# using test data
verifyShape(testData::TestDataScalar) = verifyShape(testData.data, testData)
verifyShape(testData::TestDataArray) = verifyShape(testData.array, testData)

# using data types
verifyShape(nb::Number, expected::ShapeData) = verifyShape(Shape(nb), expected)
verifyShape(tuple::Tuple, expected::ShapeData) = verifyShape(Shape(tuple), expected)
verifyShape(array::AbstractArray, expected::ShapeData) = verifyShape(Shape(array), expected)

# using ndarray
verifyShape(ndarray::NDArray, expected::ShapeData) = verifyShape(ndarray.shape, expected)

# using ragged array
verifyShape(test, expected::ShapeData, dtype) = verifyShape(Shape(test.array; dtype=dtype), expected)

function verifyShape(array1, array2)
    return size(array1) == size(array2) && ndims(array1) == ndims(array2)
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests Utils ( check array content )                                             #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function compareArrayContent(ndarray, testData::TestDataArray)
    return compareByIndex(ndarray, testData.matrix) &&
           compareByIterator(ndarray, testData.matrix) &&
           compareBySlices(ndarray, testData.matrix)
end


function compareByIndex(ndarray, expected)
    ranges = ntuple(i -> 1:size(ndarray)[i], ndims(ndarray))

    for indices in Iterators.product(ranges...)
           println("index used: ", indices)
        if (ndarray[indices...] != expected[indices...])
            println("Comparison byIndex failed at index: ", indices)
           # println(" Original was", ndarray)
           # println(" Expected was", expected)
            println("  Expected: ", expected[indices...])
            println("  Got:      ", ndarray[indices...])
            return false
        end
    end
    return true
end

function compareByIterator(ndarray, expected)
    for (got, shouldBe) in zip(ndarray, expected)
        if (got != shouldBe)
            println("Comparison byIterator failed at element ", i)
            println(" Original was", ndarray)
            println("  Expected: ", shouldBe)
            println("  Got:      ", got)
            return false
        end
    end
    return true
end

function compareBySlices(ndarray, expected)
    shape = size(ndarray)

    for axis in 1:ndims(ndarray)
        for idx in CartesianIndices(shape)

            idx_tuple = Tuple(idx)
            array = Array{Union{Int,Colon}}(undef, length(idx_tuple))

            for (i, val) in enumerate(idx_tuple)
                if (i == axis)
                    array[i] = Colon()
                else
                    array[i] = val
                end
            end

            if (!comparesSlices(ndarray, expected, array))
                return false
            end
        end

    end
    return true
end

function comparesSlices(ndarray, expected, indices)
    println("A: $ndarray")
    println("slices made: A[$(indices)]")

    slice_test::SubArray = ndarray[indices...]
    slice_expected::Array = expected[indices...]

    contentMatch = (slice_test == slice_expected)

    if !contentMatch
        println("Comparison failed on slice along with indices $indices")
        println("Ranges were $indices")
        println("Expected slice: ", slice_expected)
        println("Got slice:      ", slice_test)
    end
    return contentMatch
end


# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests Utils ( check exceptions )                                                #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#

function exceptionPassed(f::Function, nb::Int, ExceptionType::Type)
    passed = true

    try
        x = f()
        passed = false
        println("      No exception thrown, in exception: $(nb)")
    catch e
        if (isa(e, ExceptionType))
            passed = true
        else
            passed = false
            println("      Wrong exception in exception: $(nb), excepted: $ExceptionType was: $(typeof(e))")
            rethrow(e)
        end
    end
    return passed
end

function allExceptionPassed(exceptions::Base.Vector{Tuple{Function,DataType}})
    for (i, tuple) in enumerate(exceptions)
        f, exceptionType = tuple[1], tuple[2]
        exceptionPassed(f, i, exceptionType) || return false
    end
    return true
end

# ---------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                  #
#                                                  Tests Utils ( write to logs )                                                #
#                                                                                                                                  #
# ---------------------------------------------------------------------------------------------------------------------------------#
# Function to write timestamped messages to the log




function write_content(ndarray, write_log_fn::Function)
    logShape(ndarray, write_log_fn)
    logByIndex(ndarray, write_log_fn)
    logByIterator(ndarray, write_log_fn)
    logBySlices(ndarray, write_log_fn)
end


function write_content(scalar::AbstractFloat, write_log_fn::Function)
    write_log_fn(string(round(scalar, digits=7)))
end

function write_content(scalar::Number, write_log_fn::Function)
    write_log_fn(string(scalar))
end


function write_content(ndarray::NDArray{T, 0}, write_log_fn::Function) where {T}
    write_log_fn("$(ndarray[()])")
    logShape(ndarray, write_log_fn)
end

function write_content(ndarray::AbstractArray{T, 0}, write_log_fn::Function) where {T}
    write_log_fn("$(ndarray[])")
    logShape(ndarray, write_log_fn)
end

function write_content(ndarray::TestDataScalar, write_log_fn::Function)
    write_log_fn("$(ndarray.data)")
end


logShape(ndarray::AbstractArray, write_log_fn::Function) = logShape(ndims(ndarray), Base.prod(size(ndarray)), size(ndarray), write_log_fn)

logShape(ndarray::NDArray, write_log_fn::Function) = logShape(ndims(ndarray), ndarray.shape.length, size(ndarray), write_log_fn)

logShape(testData::ShapeData, write_log_fn::Function) = logShape(testData.dim, testData.total_element, testData.shape, write_log_fn)

function logShape(dim, total_element, shape, write_log_fn::Function)
    toLog = "dim: " * string(dim) * "\ntotal_element: " * string(total_element) * "\nshape: " * string(shape)
    write_log_fn(toLog)
end


logByIndex(testData::TestDataArray, write_log_fn::Function) = logByIndex(testData.matrix, write_log_fn)

function logByIndex(array::AbstractArray, write_log_fn::Function)
    ranges = ntuple(i -> 1:size(array)[i], ndims(array))
    contentByIndex = "index: "

    for indices in Iterators.product(ranges...)
        got = array[indices...]
        scalar =  typeof(got) <: AbstractFloat ? round(got, digits=7) : got
        contentByIndex *= "$(pretty_index(indices)): $scalar, "
    end
    write_log_fn(contentByIndex)
end


logByIterator(testData::TestDataArray, write_log_fn::Function) = logByIterator(testData.matrix, write_log_fn)

function logByIterator(array::AbstractArray, write_log_fn::Function)
    contentByIterator = "iterator: "
    for got in array
        s = typeof(got) <: AbstractFloat ? round(got, digits=7) : got
        contentByIterator *= "$s "
    end
    write_log_fn(contentByIterator)
end


logBySlices(testData::TestDataArray, write_log_fn::Function) = logBySlices(testData.matrix, write_log_fn)


function logBySlices(array::AbstractArray, write_log_fn::Function)
    shape = size(array)

    contentBySlices = ""

    write_log_fn("slices:")

    for axis in 1:ndims(array)
        for idx in CartesianIndices(ntuple(i -> i==axis ? 1 : 1:shape[i], ndims(array)))
        slice_idx = Tuple(idx)  # indices for the non-axis dimensions
        full_idx = ntuple(i -> i == axis ? Colon() : slice_idx[i], ndims(array))
        println("slices made: A[$(full_idx)]")
        contentBySlices *= logSlices(array, full_idx)
        end

    end
    write_log_fn(contentBySlices)
end



function round(x; digits::Int)
    if (x == -0.0)
        x = 0.0
    end
    return Base.round(x, digits=digits) 
end

function logSlices(matrix::AbstractArray{<:AbstractFloat}, indices)
    slice_test = matrix[indices...]          # get slice
    # convert each element to rounded number string
    slice_str = isempty(slice_test) ? "" : 
                "[" * join(round.(slice_test, digits=7), ", ") * "]"
    
    str = isempty(slice_test) ? 
          "array[$(pretty_index(indices...))] = []\n" :
          "array[$(pretty_index(indices...))] = $slice_str\n"
    
    return str
end

function logSlices(matrix::AbstractArray, indices)
    slice_test = matrix[indices...]
    str = isempty(slice_test) ? "array[$(pretty_index(indices...))] = []\n" : "array[$(pretty_index(indices...))] = $slice_test\n"
    return replace(str, "Any" => "")
end

end # TestUtils