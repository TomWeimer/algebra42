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

my_1d_array     =  NDArray{Int}([10, 20, 30, 40, 50])

my_2d_array     =  NDArray{Int}([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

my_3d_array     =  NDArray{Int}([
[
    [1, 2],
    [3, 4],
    [5, 6]
],

[
    [7, 8],
    [9, 10],
    [11, 12]
],

[
    [13, 14],
    [15, 16],
    [17, 18]
],

[
    [19, 20],
    [21, 22],
    [23, 24]
]])




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
    ranges = ntuple(i -> 1:size(ndarray)[i], ndims(ndarray))

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

            if (!comparesSlices(ndarray, expected, array))
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
function slicesOk(test, expected::Number, indices)
    ndarray = NDArray{Int}(test.data);
    
    f = () -> ndarray[indices...]    

    return exceptionPassed(f, 1, DomainError)
end

# 'normal' ndarray
function slicesOk(test, expected, indices)
    ndarray         = NDArray{Int}(test.data);
    ndarrayToChange = NDArray{Int}(test.data);
    
    sliced = ndarray[indices...]
    expectedSliced = expected[indices...]
    slicedToChange = ndarrayToChange[indices...]

    shapeOk   = verifyShape(sliced, expectedSliced);
    contentOk = comparesSlices(ndarray, expected, indices);
    
    if (isempty(expected))
        return shapeOk && contentOk
    end


    idx1 = nindexes(ndims(ndarray))
    idx2 = nindexes(ndims(sliced))
    mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
    passByRef = (ndarrayToChange[idx1...] == 10)


    shapeOk || println("shape mismatch")
    contentOk || println("content mismatch")
    mutableOk || println("mutable: ", mutableOk, " should be ", true) 
    passByRef || println("passByRef: ", passByRef, " should be ", true)

    return shapeOk && contentOk && mutableOk && passByRef;
end



# 'normal' ndarray
function slicesRaggedOk(test, expected, indices)
    ndarray         = NDArray{Any}(test.data);
    ndarrayToChange = NDArray{Any}(test.data);
    
    sliced = ndarray[indices...]
    expectedSliced = expected[indices...]
    slicedToChange = ndarrayToChange[indices...]

    shapeOk   = verifyShape(sliced, expectedSliced);
    contentOk = comparesSlices(ndarray, expected, indices);
    if (isempty(expected))
        return shapeOk && contentOk
    end
    idx1 = nindexes(ndims(ndarray))
    idx2 = nindexes(ndims(sliced))
    mutableOk = mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, 10)
    passByRef = (ndarrayToChange[idx1...] == 10)

    shapeOk || println("shape mismatch")
    contentOk || println("content mismatch")
    mutableOk || println("mutable: ", mutableOk, " should be ", true) 
    passByRef || println("passByRef: ", passByRef, " should be ", true)

    return shapeOk && contentOk && mutableOk && passByRef;
end

function mutateNdArray(slicedToChange, ndarrayToChange, idx1, idx2, val)
    slicedToChange[idx2...] = val;
    return ndarrayToChange[idx1...] == val
end


function testFancyIndexingBooleanMask(ndarray)
    # Test a basic boolean mask
    mask = NDArray{Bool}([true, false, true, false, true])
    selected_elements = ndarray[mask]
    
    shouldBe = selected_elements == [10, 30, 50]
    size = length(selected_elements) == 3

    return shouldBe && size;
end

function testFancyIndexingEmptyBooleanMask(ndarray)
  # Test an empty boolean mask
    empty_mask = [false, false, false, false, false]
    should =  ndarray[empty_mask] == []

    # Test a full boolean mask
    full_mask = [true, true, true, true, true]
    content =  ndarray[full_mask] == ndarray

    # Test with a different size mask (this will throw a BoundsError)
    error = exceptionPassed(() -> ndarray[[true, false]], 1, DomainError)

    return should && content && error
end

function testFancyIndexingMultiDim(my_2d_array)
    # Test with a boolean mask on the rows
    row_mask = [true, false, true]
    passed1 = compareByIndex(my_2d_array[row_mask, :], [1 2 3; 7 8 9])

    col_indices = [1, 3]
    passed2 = compareByIndex(my_2d_array[:, col_indices], [1 3; 4 6; 7 9])
    
    passed3 = compareByIndex(my_2d_array[[1, 3], [1, 3]], [1 3; 7 9])

    return passed1 && passed2 && passed3
end


function testFancyIndexing3d(my_3d_array)
        # --- Indexing on Dimension 1 ---
        dim1_mask = [true, false]
        dim1_indices = [2]
        
        passed1 = compareByIndex(my_3d_array[dim1_mask, :, :], [1 3 5 7; 9 11 13 15; 17 19 21 23])
        passed2 =  compareByIndex(my_3d_array[dim1_indices, :, :], [2 4 6 8; 10 12 14 16; 18 20 22 24])

        # --- Indexing on Dimension 2 ---
        dim2_mask = [true, false, true]
        dim2_indices = [1, 3]

        passed3 = compareByIndex(my_3d_array[:, dim2_mask, :], [1 5 9 13; 2 6 10 14])
        passed4 =  compareByIndex(my_3d_array[:, dim2_indices, :], [1 5 9 13; 2 6 10 14])

        # --- Indexing on Dimension 3 ---
        dim3_mask = [false, true, false, true]
        dim3_indices = [2, 4]
        
        passed5 =  compareByIndex(my_3d_array[:, :, dim3_mask], [3 7 11 15; 4 8 12 16])
        passed6 =   compareByIndex(my_3d_array[:, :, dim3_indices], [3 7 11 15; 4 8 12 16])

        return passed1 && passed2 && passed3 && passed4 && passed5 && passed6
    end
    
    #--------------------------------------------------------------------------------------------------------------------------

   function mixedAndCombinedIndexing(my_3d_array)
        # --- Boolean and Integer Indexing ---
        row_mask = [true, false]
        col_indices = [1, 3]
        depth_mask = [true, false, true, false]

        passed1 = compareByIndex(my_3d_array[row_mask, col_indices, depth_mask], [1 5 9 13; 2 6 10 14])

        # --- Using arrays for all dimensions ---
        row_indices = [1, 2]
        col_indices = [3, 1]
        depth_indices = [4, 2]

        passed2 = compareByIndex(my_3d_array[row_indices, col_indices, depth_indices], [16 10; 24 18])

        # --- Edge case with empty result ---
        passed3 =  isempty(my_3d_array[:, [false, false, false], :])

        return passed1 && passed2 && passed3
    end
    
    #--------------------------------------------------------------------------------------------------------------------------

    function correctnessAndMutability(my_3d_array)
        # Fancy indexing returns a copy, not a view
        fancy_slice = my_3d_array[[1], :, :]
        fancy_slice[1] = 999
        return  my_3d_array[1] != 999
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


    printstyled("\n   check slices:\n", bold=false, italic=true)
    printTest("scalar",         slicesOk(scalar, scalar.expected.content, (:,)))
    printTest("vector",         slicesOk(vector1D, vector1D.expected.content, (:,)))
    printTest("matrix",         slicesOk(matrix2D, matrix2D.expected.content, (1, :)))
    printTest("array3D",        slicesOk(array3D, array3D.expected.content, (1, 1, :)))
    printTest("emptyArray",     slicesOk(emptyArray, emptyArray.expected.content, (:,)))
    printTest("columnVector",   slicesOk(columnVector2D, columnVector2D.expected.content, (1, :)))
    printTest("rowVector",      slicesOk(rowVector2D, rowVector2D.expected.content, (1, :)))
    printTest("raggedArray",    slicesRaggedOk(raggedArray, raggedArray.expected.content, (:,)))

     printstyled("\n   check fancy indexing:\n", bold=false, italic=true)
    printTest("boolean mask",         testFancyIndexingBooleanMask(my_1d_array))
    printTest("empty boolean row_mask",         testFancyIndexingEmptyBooleanMask(my_1d_array))
    printTest("multi dim",         testFancyIndexingMultiDim(my_2d_array))
    printTest("multi dim (3d)",        testFancyIndexing3d(my_3d_array))
    printTest("mixed and combined",     mixedAndCombinedIndexing(my_3d_array))
    printTest("mutability (must copy)",   correctnessAndMutability(my_3d_array))
end

testNdArray()