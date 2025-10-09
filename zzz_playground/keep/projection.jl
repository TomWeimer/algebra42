
# Return a copy of the array with only the specified elements
const CopyIndices =  Union{AbstractRange, Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}

# Type of the elements used in fancy indexing
const FancyIndices = Union{AbstractArray{Int}, AbstractArray{Bool}}

# Acess:
#-------

# Access to fancy indexing using directly only one index
function Base.getindex(array::NDArray{DType, N}, indices::CopyIndices ) where {DType, N}
    shape = size(array)
    convertedIndices = convertIndices(shape, indices);
    return fancyIndexing(array, convertedIndices)
end

# Access to fancy indexing using directly multiple index
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{CopyIndices} ) where {DType, N}
    shape = size(array)
    convertedIndices = convertIndices(shape, indices);
    return fancyIndexing(array, convertedIndices)
end


# Process:
#---------

# A single bool mask
function fancyIndexing(array::NDArray{DType, N}, indices::Tuple{AbstractArray{Bool}})  where {DType, N}
    mask = indices[1]

    # Verify that they have the same number of elements
   length(mask) == array.shape.length || throw(DIMENSIONS_DO_NOT_MATCH)

    # Create the array if the boolean was true
    filteredArray = [ val for (i, val) in enumerate(array) if mask[i] == true ]

    # Return a new NDArray
    return NDArray{DType}(filteredArray)
end



# Multiple index
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ Vararg{ AbstractArray{Int} }} ) where {DType, N}
    
    output_shape, broadcast_function  = advanced_indexing(indices) do output_idx::CartesianIndex
        src_idx = ntuple(j -> indices[j][ output_idx[j] ], length(indices) )
        return src_idx
    end

    output_array = NDArray{DType}(output_shape)
    
    for i in CartesianIndices(output_shape)
        output_array[i] = original_array[ broadcast_function(i) ]
    end
end



# Broadcasting advanced indexing:
# -------------------------------

function advanced_indexing(indices::Tuple{Vararg{AbstractArray{Int}}}, f::Function)
# Broadcasting can be understood by four rules
#   1. All input arrays with ndim smaller than the input array of largest ndim have
#      1’s pre-pended to their shapes.
#
#   2. The size in each dimension of the output shape is the maximum of all the
#      input shapes in that dimension.
#
#   3. An input can be used in the calculation if it’s shape in a particular dimension
#      either matches the output shape or has value exactly 1.
#
#   4. If an input has a dimension size of 1 in its shape, the first data entry in that
#      dimension will be used for all calculations along that dimension. In other
#      words, the stepping machinery of the ufunc will simply not step along that
#      dimension when otherwise needed (the stride will be 0 for that dimension).

    # Step 1: Prepend input arrays with ones
    
    padded_shapes = prependInputArraysShape(indices)

    # Step 2: Obtain output shape

    N = length(padded_shapes[1])

    output_shape = obtain_output_shape(padded_shapes, N)

    # Step 3: Either the input can be used as a "broadcast constant" or we use it as a "variable index"
    is_broadcast_compatible(output_shape, padded_shapes, N) || throw(DomainError("The indices entered are not compatible for broadcasting"))

    # Step 4: Use broadcast constant constant when shape is 1 in a dimension
    broadcast_constant_index, broadcast_constant_values = obtain_broadcast_constant(padded_shapes::AbstractArray{PaddedShape}, indices, N::Int)

    # Create and return a function to be broadcast on the original_array 
    return output_shape, function projected_f(input_index::CartesianIndex)
        # Create a full CartesianIndex for the original function
        output_index = zeros(Int, N)

        # Fill the output index with the broadcast constant
        for i in 1:eachindex(broadcast_constant_index)
            output_index[ broadcast_constant_index[i] ] = broadcast_constant_values[i]
        end

        # Fill the output index with the non constant values
        for (i, val) in enumerate(output_index)
            if (val == 0) 
                output_index[i] = input_index[i] 
            end
        end
        
        # Call the original function with the full CartesianIndex
        return f(CartesianIndex(Tuple(output_index)))
    end
end

# Step 1
function prependInputArraysShape(indices::Tuple{Vararg{AbstractArray{Int}}})
    maxDim = 0
    return [ PaddedShape(index, maxDim) for index in indices]
end

# Step 2
function obtain_output_shape(padded_shapes::AbstractArray{PaddedShape}, N::Int)
    return ntuple(i -> maximum(ps[i] for ps in padded_shapes), N)
end

# Step 3
function is_broadcast_compatible(output_shape::Tuple, padded_shapes::AbstractArray{PaddedShape}, N::Int)
    return all(i -> all(ps[i] == 1 || ps[1] == output_shape[i] for ps in padded_shapes), 1:N)
end

# Step 4
function obtain_broadcast_constant(padded_shapes::AbstractArray{PaddedShape}, original_indices::Tuple{Vararg{AbstractArray{Int}}}, N::Int)
    broadcast_constant_index  = []
    broadcast_constant_values = []

    for i in 1:N
        len_dim = length(original_indices[i])
        for j in 1:len_dim
            if padded_shapes[i][j] == 1
                push!(broadcast_constant_index, (i, j))
                push!(broadcast_constant_values, original_indices[i][j])
            end
        end
    end
    return broadcast_constant_index, broadcast_constant_values
end

function convertIndices(shape::Tuple, indices::Tuple{Vararg{CopyIndices}})
    return ntuple(i -> convertIndex(shape, i, indices[i]), length(indices))
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Bool})
    return findall(index);
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Int})
    return index;
end

function convertIndex(shape::Tuple, i::Int, index::Int)
    return [index];
end

function convertIndex(shape::Tuple, i::Int, index::Colon)
    return [x for x in 1:shape[i]];
end

function convertIndex(shape::Shape, i::Int, index::AbstractRange)
    return collect(index)
end


# Padded Shape:
# ------------------

# Structure used when prepending shapes with ones
struct PaddedShape{OriginalDimension, ExpectedDimension}
    originalShape::Ref{NTuple{OriginalDimension, Ref{Int}}}
end

function PaddedShape(originalShape::NTuple{N, Int}, maxDim::Ref{Int}) where {N}
    # If the max dimension is smaller than the dimension of original shape then update it
    if ( maxDim[] < N)
        maxDim[] = N
    end
    # Return a "view" on the original shape
    PaddedShape{N, maxDim}( Ref(originalShape) )
end

function PaddedShape(originalArray::AbstractArray{T, N}, maxDim::Ref{Int}) where {T, N}
    # If the max dimension is smaller than the dimension of the original array then update it
    ( maxDim[] < N) && maxDim[] = N
    # Return a "view" on the shape of the array
    PaddedShape{N, maxDim}( Ref(size(originalArray) ))
end

# Using this function and an offset we mimic a shape prepended with 1 without allocating it
function getindex(prep_shape::PaddedShape{OriginalDimension, ExpectedDimension}, i::Int) where {OriginalDimension, ExpectedDimension}
    offset = ExpectedDimension[] - OriginalDimension
    return (i <= offset) ? 1 : prep_shape.originalShape[][i - offset]
end

# Return the original dimension of the shape
original_dim(padded_shape::PaddedShape{OriginalDimension, ExpectedDimension}) where {OriginalDimension, ExpectedDimension} = OriginalDimension

Base.length(padded_shape::PaddedShape{OriginalDimension, ExpectedDimension}) where {OriginalDimension, ExpectedDimension} = ExpectedDimension[] 