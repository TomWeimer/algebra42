
using Base.Broadcast: broadcast_shape

const Indices = Union{AbstractRange, AbstractArray{Int}, AbstractArray{Bool}, Colon, Int}
const FancyIndices = Union{AbstractArray{Int}, AbstractArray{Bool}}




# Exceptions:
# ----------------

const DIMENSIONS_DO_NOT_MATCH    =  DomainError("The dimensions do not match ")
const INDICES_OUT_OF_BOUNDS      =  DomainError("The indices entered are outof bounds")
const SCALAR_ONLY_ACCESS_ERROR   =  DomainError("Cannot access non 0 dimensional array wiht array[()]")
const TYPE_NOT_ACCEPTED_AS_INDEX =  DomainError("The type cannot be used to access elements in NDArray")



# Fancy indexing:
# ---------------

function Base.getindex(array::NDArray{DType, N}, indices ) where {DType, N}
    shape = size(array)
    indicesOutOfBounds(shape, indices)(shape, indices)
    return fancyIndexing(array, indices)
end



# Fancy indexing:
# ---------------


# A single bool mask
function fancyIndexing(array::NDArray{DType, N}, indices::Tuple{Vector{Bool}} )
    # Obtain the mask
    mask = indices[1];

    # Verify that they have the same number of elements
   shapesMatch(array, mask) || throw(DIMENSIONS_DO_NOT_MATCH)

    # Create the array if the boolean was true
    filteredArray = [ val for (i, val) in enumerate(array) if mask[i] == true ]

    # Return a new NDArray
    return ndarray{DType}(filteredArray)
end

# Fancy indexing (with only vector)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ AbstractVector{Bool}, Vararg{ AbstractVector{FancyIndices} }} ) where {DType, N, T}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: the shape of the indices must be compatible

        
    verifyFancyIndexing(padded_shapes, length(final_shape) )


    final_shape = getFinalShape(indices)

    expandedIndices = expandDimensions( size(original_array), indices )

    output_array = ndarray{DType}(final_shape)

    source_grid = CartesianIndices(original_array)[expandedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end

# Fancy indexing (with multi-dimensional array int)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ Vararg{ AbstractArray{Bool, M} }}  ) where {DType, N, M}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: very strange try
    # > A =  [1 2 3; 4 5 6; 7 8 9]
    # > I1 = [ 1 1; 2 2]
    # > I2 = [1 2; 3 1]
    # It returns somehow a 2x2x2x2 matrix
    # So I choose to implement numpy way

    all(x -> size(x) == indices[1], indices) || throw( DomainError("Size should be the same") )

    final_shape = getFinalShape(indices);
    
    # verifyFancyIndexing(padded_shapes, length(final_shape) )
    
    expandedIndices = expandDimensions( size(original_array), indices )

    output_array = ndarray{DType}(final_shape)

    source_grid = CartesianIndices(original_array)[expandedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end


"""
    check_broadcast_compatible(shapes::Vararg{Tuple})

Check if multiple shapes are broadcast-compatible. Returns true if compatible, false otherwise.
"""
function check_broadcast_compatible(shapes::Vararg{Tuple})
    if length(shapes) == 0
        return true
    end
    
    # Find the maximum number of dimensions
    max_ndims = maximum(length.(shapes))
    
    # Compare each dimension starting from the last
    for i in 1:max_ndims
        dims = Int[]
        for s in shapes
            # Pick dimension from the right, default 1 if missing
            push!(dims, i <= length(s) ? s[end-i+1] : 1)
        end
        # Check if all are equal or 1
        max_dim = maximum(dims)
        for d in dims
            if d != 1 && d != max_dim
                return false
            end
        end
    end
    
    return true
end



function advanced_index_shape(array_shape::NTuple, indices...)
    remaining_axes = collect(array_shape)
    
    # Track broadcasted shape manually
    broadcasted_shape = ()
    
    for (axis, idx) in enumerate(indices)
        if eltype(idx) <: Bool
            # Boolean array: replace axis with number of true values
            n_true = count(x -> x, idx)
            if isempty(broadcasted_shape)
                broadcasted_shape = (n_true,)
            else
                # Manual broadcasting: combine shapes elementwise
                new_shape = size(idx)
                max_len = max(length(broadcasted_shape), length(new_shape))
                broadcasted_shape = tuple([i <= length(broadcasted_shape) ? broadcasted_shape[i] : 1 for i in 1:max_len]... )
                new_shape = tuple([i <= length(new_shape) ? new_shape[i] : 1 for i in 1:max_len]... )
                broadcasted_shape = tuple([max(broadcasted_shape[i], new_shape[i]) for i in 1:max_len]...)
            end
            remaining_axes[axis] = 1
        else
            # Integer array: replace axis with its shape (broadcast manually)
            new_shape = size(idx)
            if isempty(broadcasted_shape)
                broadcasted_shape = new_shape
            else
                max_len = max(length(broadcasted_shape), length(new_shape))
                broadcasted_shape = tuple([i <= length(broadcasted_shape) ? broadcasted_shape[i] : 1 for i in 1:max_len]... )
                new_shape = tuple([i <= length(new_shape) ? new_shape[i] : 1 for i in 1:max_len]... )
                broadcasted_shape = tuple([max(broadcasted_shape[i], new_shape[i]) for i in 1:max_len]...)
            end
            remaining_axes[axis] = 1
        end
    end
    
    # Append remaining axes
    final_shape = tuple(broadcasted_shape..., remaining_axes[length(indices)+1:end]...)
    return final_shape
end


# Fancy indexing (with multi-dimensional array bool)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ AbstractArray{Int}, Vararg{ AbstractArray{FancyIndices, M} }}  ) where {DType, N, M}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: very strange try
    # > A =  [1 2 3; 4 5 6; 7 8 9]
    # > I1 = [ 1 1; 2 2]
    # > I2 = [1 2; 3 1]
    # It returns somehow a 2x2x2x2 matrix
    # So I choose to implement numpy way

    all(x -> size(x) == indices[1], indices) || throw( DomainError("Size should be the same") )

    final_shape = advanced_index_shape(size(original_array), indices...);
    
    check_broadcast_compatible(size.(indices)) || throw( DomainError("Shapes are not compatible"))
    
    reducedIndices = reduceIndices(generatedIndices, final_shape,  size(original_array))

    output_array = ndarray{DType}(final_shape)

    source_grid = CartesianIndices(final_shape)[reducedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end


# Fancy indexing (with multi-dimensional array bool)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ AbstractArray{Bool}, Vararg{ AbstractArray{FancyIndices, M} }}  ) where {DType, N, M}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: very strange try
    # > A =  [1 2 3; 4 5 6; 7 8 9]
    # > I1 = [ 1 1; 2 2]
    # > I2 = [1 2; 3 1]
    # It returns somehow a 2x2x2x2 matrix
    # So I choose to implement numpy way

    all(x -> size(x) == indices[1], indices) || throw( DomainError("Size should be the same") )

    final_shape = advanced_index_shape(size(original_array), indices...);

    check_broadcast_compatible(size.(indices))
    
    expandedIndices = reduceIndices(generatedIndices, size(original_array), final_shape )

    output_array = ndarray{DType}(final_shape)

    source_grid = CartesianIndices(original_array)[expandedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end


function getPaddedShape(args::Tuple{Vararg{Vector{FancyIndices}}})

end

getFinalShape(args::Tuple{Vararg{Vector{FancyIndices}}}) = map(getDim, args)

getFinalShape(args::Tuple{Vararg{AbstractArray{FancyIndices}}}) = map(getDim, args)

getDim(arg::Vector{Bool}) = count(arg)

getDim(arg::Vector{Int})  = maximum(arg)


getFinalShape(args::Tuple{ Vararg{ AbstractArray{FancyIndices} }}) = map(getDimension, args)


function getDimension( a::AbstractArray{Int} )
    
end

function getDimension(a::AbstractArray{Bool})
end


function getFinalShape(args::NTuple{N, Vector{Bool}}, shape) where {N}
    # This generator both transforms the input (e.g., gets the count)
    # and filters out the zero values with the `if d > 0` clause.
    dims = ntuple(count, args)

    # We then filter out the zero values before creating the tuple.
    return tuple(d for d in dims if d > 0)
end


function elementIsTaken(element::Union{AbstractArray{Bool}, Colon})
    return isa(element, Colon) || element == true
end


function shapesMatch(ndarray::NDArray{DType, N}, array::Vector{T}) where {DType, N, T}
    return ndarray.shape.length == length(array)
end

function shapesMatch(ndarray::NDArray{DType, N}, indices::NTuple{N, Vector{Bool}}) where {DType, N}
    shape = size(ndarray)
    return all( (index, dim) -> length(index) == dim, zip(indices, shape))
end

function shapesMatch(ndarray::NDArray{DType, N}, indices::NTuple{N, Vector{Int}}) where {DType, N}
    shape = size(ndarray)
    return all( (index, dim) -> length(index) <= dim, zip(indices, shape))
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


function reduceIndices(generatedIndices::Tuple{Vararg{T}}, originalShape, actualShape) where {T}
    dimIndices, dimActualShape, dimOriginalShape = size(indices), length(actualShape), length(originalShape);
    
    return ntuple(i -> i <= dimActualShape ? generatedIndices[i] : 0, originalShape)
end





# Helper Function:
# ----------------

function verifyAndExpandIndices(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{T}}) where {T}
    # Check that the indices are valid, or throw an exception
    indicesOutOfBounds(shape, indices)

    # Expand the dimensions, or throw the an exception
    return expandDimensions(shape, indices)
end

function indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Int}}})
    all( isBetween.(array, 1, dim) for (array, dim) in zip(indices, shape) ) || throw( INDICES_OUT_OF_BOUNDS )
end

indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Bool}}})  =  false;

function isBetween(number::Real, minIncl::Real, maxIncl::Real) 
    return minIncl <= number <= maxIncl
end


function verifyFancyIndexing(padded_shapes, max_ndim)
   for dim in 1:max_ndim
      # Collect the size of each array along this dimension
      dim_sizes = [shape[dim] for shape in padded_shapes]

      # Ignore trivial size-1 dimensions (they can broadcast)
      nontrivial_sizes = filter(!=(1), dim_sizes)

      # If more than one distinct size remains, broadcasting fails
      if length(unique(nontrivial_sizes)) > 1
         throw(DomainError("Broadcast shape mismatch along dimension $dim"))
      end
   end
end

# This function is broadcast on CartesianIndices
function project_index_function(cartesianIdx::CartesianIndex, shape::NTuple{N, Int}) where {N}
   # The cartesianIdx is generated by CartesianIndices(final_shape) which means that it is one possible index of the output array
   # We need to return with this function an index associated to the original array
   # From the combination of each shape and cartesian index we create one possible index

   # We check that the i-th dimension was not stretched ( shape[i] == 1), if it was the case then the only possible index is 1
   # Otherwise we return the contained in the index
   return CartesianIndex(ntuple( i -> (shape[i] == 1) ? 1 : cartesianIdx[i], length(shape)))
end


function fancyIndexing(array::NDArray{DType, N}, masks::NTuple{N, Vector{Int}} ) where {DType, N, T}

    noColon = filter(x -> !isa(x, Colon), masks);
    # Step 1: pad shapes
    max_ndim = maximum(ndims.(noColon))
    
    padded_shapes = [
        ntuple(i -> i <= ndims(mask) ? size(mask, i) : 1, max_ndim) for mask in noColon
    ]

    # Step 2: final broadcasted shape
    final_shape = map(maximum, zip(padded_shapes...))

    # Step 3: verify
    verifyFancyIndexing(padded_shapes, max_ndim)

    # Step 4: allocate output
    output = ndarray{DType}(final_shape)

    # Step 5: loop
    for idx in CartesianIndices(final_shape)
        # Compute the projected indices for each mask
        projected_indices = [project_index_function(idx, padded_shapes[j]) for j in 1:length(masks)]

        # Look up the actual integer indices from the masks
        real_indices = map(j -> masks[j][projected_indices[j]], 1:length(masks))

        # Use splatting to index into your NDArray
        output[idx] = array[real_indices...]
    end

    return output
end