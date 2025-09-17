
# Return a copy of the array with only the specified elements
const CopyIndices =  Union{AbstractRange, Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}

# Type of the elements used in fancy indexing
const FancyIndices = Union{AbstractArray{Int}, AbstractArray{Bool}}

# Exceptions:
# ----------------

const DIMENSIONS_DO_NOT_MATCH    =  DomainError("The dimensions do not match ")
const INDICES_OUT_OF_BOUNDS      =  DomainError("The indices entered are outof bounds")
const SCALAR_ONLY_ACCESS_ERROR   =  DomainError("Cannot access non 0 dimensional array wiht array[()]")
const TYPE_NOT_ACCEPTED_AS_INDEX =  DomainError("The type cannot be used to access elements in NDArray")



# Fancy indexing:
# ---------------

function Base.getindex(array::NDArray{DType, N}, indices::CopyIndices ) where {DType, N}
    shape = size(array)
    convertedIndices = convertIndices(shape, (indices, ));
    return fancyIndexing(array, convertedIndices)
end

function Base.getindex(array::NDArray{DType, N}, indices::Vararg{CopyIndices} ) where {DType, N}
    shape = size(array)
    convertedIndices = convertIndices(shape, indices);
    return fancyIndexing(array, convertedIndices)
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
    println("shape: ", shape)
    return [x for x in 1:shape[i]];
end

function convertIndex(shape::Shape, i::Int, index::AbstractRange)
    return collect(index)
end


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

# Fancy indexing (with only array of bool or int)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ Vararg{ FancyIndices }} ) where {DType, N}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: the shape of the indices must be compatible
    original_shape = size(original_array)


    println("indices: ", indices, " enter: ", all(index ->  length(size(index)) == 1, indices))
  
    final_shape = all(index ->  length(size(index)) == 1, indices) ?  ntuple(i -> length(indices[i]), length(indices)) : advanced_index_shape(original_shape, indices)

    println("final_shape: ", final_shape)
        
    check_broadcast_compatible(indices) || throw(DIMENSIONS_DO_NOT_MATCH)

    expandedIndices = expandDimensions( original_shape, indices )

    output_array = NDArray{DType}(final_shape)

    source_grid = CartesianIndices(original_array)[expandedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end


function padded_shape(shape, max_len)
    # Return a new tupple with the old content and padded with 1 until max_len if needed
    return tuple([ i <= max_len - length(shape) ? 1 : shape[i - (max_len - length(shape) )]  for i in 1:max_len]... )
end


function broadcastShape(new_shape, broadcasted_shape)
    # Obtain biggest dimension

    println("new_shape: ", new_shape,"broadcasted_shape: ",  broadcasted_shape)
    max_len = max(length(broadcasted_shape), length(new_shape))

    # Padd the shape with one until to have the same dimensions
    broadcasted_shape = padded_shape(broadcasted_shape, max_len)
    new_shape = padded_shape(new_shape, max_len)

    println("age: ", new_shape,"broadcasted_shape: ",  broadcasted_shape)

    # Return the shape containing the max dim in each axis
    return tuple([max(broadcasted_shape[i], new_shape[i]) for i in 1:max_len]...)
end

"""
    check_broadcast_compatible(shapes::Vararg{Tuple})

Check if multiple shapes are broadcast-compatible. Returns true if compatible, false otherwise.
"""
function check_broadcast_compatible(shapes::Vararg{Tuple})
    isempty(shapes) && return true

    max_ndims = maximum(length.(shapes))
    return all(1:max_ndims) do i
        dims = (i <= length(s) ? s[end - i + 1] : 1 for s in shapes)
        maxdim = maximum(dims)
        all(d -> d == 1 || d == maxdim, dims)
    end
end

function check_broadcast_compatible(shapes::Tuple{Vararg{AbstractArray{FancyIndices}}})
    return check_broadcast_compatible(size.(shapes)...)
end

function check_broadcast_compatible(shapes::Vararg{Tuple})
    isempty(shapes) && return true
    max_ndims = maximum(length.(shapes))
    return all(1:max_ndims) do i
        dims = (i <= length(s) ? s[end - i + 1] : 1 for s in shapes)
        maxdim = maximum(dims)
        all(d -> d == 1 || d == maxdim, dims)
    end
end

function advanced_index_shape(array_shape::NTuple, indices::Vararg{FancyIndices})
    return advanced_index_shape(array_shape, indices)
end

function advanced_index_shape(array_shape::NTuple, indices::Tuple{Vararg{FancyIndices}})
    nd = length(array_shape)
    n_indices = length(indices)
    
    # Track broadcasted shape manually
    broadcasted_shape = ()

    for (axis, idx) in enumerate(indices)
        if eltype(idx) <: Bool
            # Boolean array → collapse axis into count
            shape_idx = count(idx, dims=axis)
        else
            # Integer array → contributes its shape
            shape_idx = size(idx)
        end
        broadcasted_shape = isempty(broadcasted_shape) ? shape_idx : broadcastShape(broadcasted_shape, shape_idx)
    end

    # Append untouched trailing axes
    remaining = n_indices < nd ? array_shape[(n_indices+1):end] : ()
    final_shape = tuple(broadcasted_shape..., remaining...)
    
    return final_shape
end

function indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Int}}})
    all( isBetween.(array, 1, dim) for (array, dim) in zip(indices, shape) ) || throw( INDICES_OUT_OF_BOUNDS )
end


indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::AbstractArray{Bool})  =  false;

indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Colon)  =  false;

indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Bool}}})  =  false;

function isBetween(number::Real, minIncl::Real, maxIncl::Real) 
    return minIncl <= number <= maxIncl
end


function expandDimensions(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{FancyIndices}})
    dimShape, dimIndices = length(shape), length(indices)

    # If the dimension of the indices is bigger throw an exception
    dimIndices <= dimShape || throw(DIMENSIONS_DO_NOT_MATCH)


    # If the dimension of the indices is smaller than the shape, then expand it
    if (dimIndices < dimShape)
        indicesExpanded = expandIndices(indices, dimShape, dimIndices)
    else
        indicesExpanded = indices 
    end

    # Returns the expanded indices
    return indicesExpanded;
end


function expandDimensions(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{T}}) where {T}
    dimShape, dimIndices = length(shape), length(indices)

    # If the dimension of the indices is bigger throw an exception
    dimIndices <= dimShape || throw(DIMENSIONS_DO_NOT_MATCH)


    # If the dimension of the indices is smaller than the shape, then expand it
    if (dimIndices < dimShape)
        indicesExpanded = expandIndices(indices, dimShape, dimIndices)
    else
        indicesExpanded = indices 
    end

    # Returns the expanded indices
    return indicesExpanded;
end

function expandIndices(indices::Tuple{Vararg{T}}, dimShape::Int, dimIndices::Int) where {T}
    return ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimShape)
end
