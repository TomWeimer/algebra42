using Algebra42

struct NDArrayResult
    content
end

struct NDArrayData
    data
    expected::NDArrayResult

    NDArrayData(data, content) = new(data, NDArrayResult(content))
end

scalar          = NDArrayData(42, 42)

vector1D        = NDArrayData([1, 2, 3, 4], [1, 2, 3, 4])


julia_matrix_3x1 = Array{Int64, 2}(undef, 3, 1)

# Now we can fill the matrix with values
julia_matrix_3x1[1, 1] = 1
julia_matrix_3x1[2, 1] = 2
julia_matrix_3x1[3, 1] = 3

julia_matrix_1x3 = Array{Int64, 2}(undef, 1, 3)

# Now we can fill the matrix with values
julia_matrix_1x3[1, 1] = 1
julia_matrix_1x3[1, 2] = 2
julia_matrix_1x3[1, 3] = 3

matrix2D        = NDArrayData([[1, 2, 3], [4, 5, 6]], [1 2 3 ; 4 5 6])

B = [1 4; 2 5; 3 6]

matrix3x2        = NDArrayData([[1, 4], [2, 5], [3, 6]], B)

columnVector2D  = NDArrayData([[1], [2], [3]], julia_matrix_3x1)

rowVector2D     = NDArrayData( [[1, 2, 3]], julia_matrix_1x3 )

# Create the two "slices" and concatenate them
A = cat([1 3; 5 7], [2 4; 6 8], dims=3)
BE = [1 2; 3 4 ;;; 5 6; 7 8];
array3D         = NDArrayData( [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], A)

array4D         = NDArrayData( (2, 3, 4, 5), Array{Int}(undef, 2, 3, 4, 5) )

raggedArray     = NDArrayData( [[1, 2], [3, 4, 5]], [[1, 2], [3, 4, 5]])

emptyArray      = NDArrayData( [], [])




function verifyShape(ndarray, expectedContent::Number)
    shapeMatch =     ( size(ndarray) == (0,) )
    dimensionMatch = ( ndims(ndarray) == 0 )

    shapeMatch  || println("Shape mismatch: got $(size(ndarray)), expected $(Base.size(expectedContent))")
    dimensionMatch || println("Dim mismatch: got   $(ndims(ndarray)), expected 0")

    return shapeMatch && dimensionMatch
end


function verifyShape(ndarray, expectedContent)

    shapeMatch =     ( size(ndarray) == Base.size(expectedContent) )
    dimensionMatch = ( ndims(ndarray) == ndims(expectedContent)   )

    shapeMatch  || println("Shape mismatch:  got $(size(ndarray)),  expected $(Base.size(expectedContent))")
    dimensionMatch || println("Dim mismatch:    got $(ndims(ndarray)), expected $(ndims(expectedContent))")
    
    return shapeMatch && dimensionMatch
end

# emptyArray
function verifyEmptyNDArray(test, expected::Collection)
    ndarray = NDArray(test.data);
   
    f1 = () ->
        for x in ndarray
            print(x)
        end

    f2 = () -> x = ndarray[1]
    f3 = () -> x = ndarray[2]
    f4 = () -> x = ndarray[1, :]

    exceptionPassed(f1, 1, DomainError) &&
    exceptionPassed(f2, 2, DomainError) &&
    exceptionPassed(f3, 1, DomainError) &&
    exceptionPassed(f4, 2, DomainError) &&
    verifyShape(ndarray, expected) && nothingHappenInIteration(ndarray);
end


function nothingHappenInIteration(emptyNdArray)
    nothingHappen = true
    
    for x in emptyNdArray
        nothingHappen = false
        break
    end

    nothingHappen || println("something happend in iteration")
    return nothingHappen
end


# scalar
function verifyNDArray(test, expected::Number)
    ndarray = NDArray{Int}(test.data);

    contentOK = (ndarray[()] == expected)
    contentOK || println("Scalar mismatch: got $(ndarray[()]), expected $(expected)")

    return  exceptionPassed(() -> Base.iterate(ndarray), 1, DomainError) &&
            exceptionPassed(() -> ndarray[1], 2, DomainError) &&
            verifyShape(ndarray, expected) && contentOK
end


# 'normal' ndarray
function verifyNDArray(test, expected)
    ndarray = NDArray{Int}(test.data);
    passed =  verifyShape(ndarray, expected) && compareArrayContent(ndarray, expected);
    passed ||  println("test: \n", ndarray, "\nexpected: \n", expected)
    return passed
end


# 'jagged' ndarray
function verifyJaggedArray(test, expected::Collection)
    ndarray = NDArray{Any}(test.data);

    f1 = () -> x = NDArray{Int}(test.data);
    errorThrown = exceptionPassed(f1, 1, DomainError)
    
    return  errorThrown && verifyShape(ndarray, expected) && compareArrayContent(ndarray, expected);
end


function compareArrayContent(ndarray, expected)
    return compareByIndex(ndarray, expected)    && 
           compareByIterator(ndarray, expected) && 
           compareBySlices(ndarray, expected);
end


function compareByIndex(ndarray, expected)
    ranges = ntuple(i -> 1:ndarray.shape.dims[i], ndims(ndarray.shape))

    for indices in Iterators.product(ranges...)
         if (ndarray[indices...] != expected[indices...])
            println("Comparison byIndex failed at index: ", indices)
            println(" Original was", ndarray)
            println("  Expected: ", expected[indices...])
            println("  Got:      ", ndarray[indices...])
            return false;
        end
    end
    return true;
end

function compareByIterator(ndarray, expected)
    for (got, shouldBe) in zip(ndarray, expected)
         if (got != shouldBe)
            println("Comparison byIterator failed at element ", i)
            println(" Original was", ndarray)
            println("  Expected: ", shouldBe)
            println("  Got:      ", got)
            return false;
        end
    end
    return true;
end



function compareBySlices(ndarray, expected)
    shape = size(ndarray)

    for axis in 1:ndims(ndarray)
        for idx in CartesianIndices(shape)

            idx_tuple = Tuple(idx)
            array = Array{Union{Int, Colon}}(undef, length(idx_tuple))

            for (i, val) in enumerate(idx_tuple)
                if (i == axis)
                    array[i] = Colon()
                else
                    array[i] = val
                end
            end

            if (!comparesSlices(ndarray, expected, indices))
                return false;
            end
        end

    end
    return true
end

function comparesSlices(ndarray, expected, indices) 
    slice_test::SubArray = ndarray[indices...]
    slice_expected::Array = expected[indices...]

    contentMatch = (slice_test == slice_expected) 

    if !contentMatch
        println("Comparison failed on slice along axis $axis with indices $full_idx")
        println("Ranges were $idx_ranges")
        println("Expected slice: ", slice_expected)
        println("Got slice:      ", slice_test)
    end
    return contentMatch
end

# 'scalar' ndarray
function slicesOk(test::Number, expected, indices)
    ndarray = NDArray{Int}(test.data);
    
    f = () -> ndarray[indices...]    
    return exceptionPassed(f, 1, DomainError)
end

# 'normal' ndarray
function slicesOk(test, expected, indices)
    ndarray = NDArray{Int}(test.data);
    sliced = ndarray[indices...]

    shapeOk =  verifyShape(ndarray, expected) && compareArrayContent(ndarray, expected);
    contentOk || println("test: \n", ndarray, "\nexpected: \n", expected)
    return passed
end

function testNdArray()
    println("NDArray Test: ")

    printTest("scalar",         verifyNDArray(scalar, scalar.expected.content))
    printTest("vector",         verifyNDArray(vector1D, vector1D.expected.content))
    printTest("matrix",         verifyNDArray(matrix2D, matrix2D.expected.content))
    printTest("matrix3x2",      verifyNDArray(matrix3x2, matrix3x2.expected.content))
    printTest("array3D",        verifyNDArray(array3D, array3D.expected.content))
    printTest("emptyArray",     verifyNDArray(emptyArray, emptyArray.expected.content))
    printTest("columnVector",   verifyNDArray(columnVector2D, columnVector2D.expected.content))
    printTest("rowVector",      verifyNDArray(rowVector2D, rowVector2D.expected.content))
    printTest("raggedArray",    verifyJaggedArray(raggedArray, raggedArray.expected.content))


    printstyled("   check slices:", bold=false, italics=true)

    printTest("scalar",         slicesOk(scalar, scalar.expected.content, (:)))

end

testNdArray()