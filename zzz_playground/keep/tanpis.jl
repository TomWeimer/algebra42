const Indices = Union{AbstractRange, AbstractArray{Int}, AbstractArray{Bool}, Colon, Int}
const FancyIndices = Union{AbstractRange, AbstractArray{Int}, AbstractArray{Bool}, Colon, Int}




# Exceptions:
# ----------------

const DIMENSIONS_DO_NOT_MATCH    =  DomainError("The dimensions do not match ")
const INDICES_OUT_OF_BOUNDS      =  DomainError("The indices entered are outof bounds")
const SCALAR_ONLY_ACCESS_ERROR   =  DomainError("Cannot access non 0 dimensional array wiht array[()]")
const TYPE_NOT_ACCEPTED_AS_INDEX =  DomainError("The type cannot be used to access elements in NDArray")

# Access to scalar:
# ----------------

# By tuple
Base.getindex(array::NDArray{DType, 0}, ::Tuple{}) where {DType} = item(array)

# By Index
Base.getindex(array::NDArray{DType, 0}, ::Any) where {DType} = throw( TYPE_NOT_ACCEPTED_AS_INDEX )


# Access Error:
# ----------------

Base.getindex(array::NDArray{DType, N}, ::Tuple{}) where {DType, N} = throw( SCALAR_ONLY_ACCESS_ERROR )

Base.getindex(array::NDArray{DType, N}, :: NTuple{N}) where {DType, N} = throw( TYPE_NOT_ACCEPTED_AS_INDEX )


# Standard Access: 
# ----------------

function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Int} ) where {DType, N}
    shape = size(array)
    indicesExpanded = verifyAndExpandIndices(shape, indices)
   return array.content[_offset(array, indicesExpanded...)];
end

function Base.getindex(array::NDArray{DType, N}, ::Vararg{Colon} ) where {DType, N}
    indicesExpanded = ntuple(i -> Colon(), ndims(array) )

    # Return a view on the full array
    return @view array[indicesExpanded...];
end

function Base.getindex(array::NDArray{DType, N}, indices::Vararg{AbstractRange} ) where {DType, N}
    shape = size(array)
    
    indicesExpanded = verifyAndExpandIndices(shape, indices)

end


function Base.getindex(array::NDArray{DType, N}, indices::Vararg{FancyIndices} ) where {DType, N}
    shape = size(array)
    indicesExpanded = verifyAndExpandIndices(shape, indices)
    return fancyInexing(array, shape, indicesExpanded)
end


# Helper Function:
# ----------------

function verifyAndExpandIndices(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{T}}) where {T}
    # Check that the indices are valid, or throw an exception
    indicesOutOfBounds(shape, indices)

    # Expand the dimensions, or throw the an exception
    return expandDimensions(shape, indices)
end

function indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{Int}})
    all( (index, dim) -> isBetween(index, 1, dim), zip(indices, shape) ) || throw( INDICES_OUT_OF_BOUNDS )
end

function indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractRange}})
    all( (r, dim) -> 1 <= first(r) <= last(r) <= dim, zip(indices, shape) ) || throw( INDICES_OUT_OF_BOUNDS )
end

function indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Int}}})
    all( isBetween.(array, 1, dim) for (array, dim) in zip(indices, shape) ) || throw( INDICES_OUT_OF_BOUNDS )
end

indicesOutOfBounds(shape::Tuple{Vararg{Int}}, indices::Tuple{Vararg{AbstractArray{Bool}}})  =  false;

function isBetween(number::Real, minIncl::Real, maxIncl::Real) 
    return minIncl <= number <= maxIncl
end


function fancyIndexing(array::NDArray{DType, N}, indices::Tuple{Vector{Bool}} )
    # Obtain the mask
    mask = indices[1];

    # Verify that they have the same number of elements
   shapesMatch(array, mask) || throw(DIMENSIONS_DO_NOT_MATCH)

    # Create the array if the boolean was true
    filteredArray = [ val for (i, val) in enumerate(array) if elementIsTaken(mask[i]) ]

    # Return a new NDArray
    return NDArray{DType}(filteredArray)
end

function getFinalShape(args::NTuple{N, Union{ Vector{Bool}, Colon}}, shape) where {N}
   finalShape = Vector{Int}()

   for (i, boolArray) in enumerate(args)

        if (isa(boolArray, Colon))
            push!(finalShape, shape[i])
      else
         numberOfTrue = length(filter(x -> x == true, boolArray))         
         if (numberOfTrue != 0)
            push!(finalShape, numberOfTrue)
         end
      end
   end
   return tuple(finalShape)
end

function create_n_dim_array(dims::Integer...)
    # Get the dimensions as a tuple
    dims_tuple = dims

    # Create an uninitialized array of the correct size
    A = Array{Float64}(undef, dims_tuple)

    # Use a for loop to populate the array
    # CartesianIndices iterates over all combinations of indices
    for I in CartesianIndices(A)
        # Your logic here to fill the array
        # This example sums the indices
        A[I] = sum(Tuple(I))
    end
    return A
end



function fancyIndexing(array::NDArray{DType, N}, indices::NTuple{N, Union{Vector{Bool}, Colon}} ) where {DType, N, T}
    
    noColon = filter(x -> !isa(x, Colon), masks);
    
    # Step 1: pad shapes
    max_ndim = maximum(ndims.(noColon))
    
    padded_shapes = [
        ntuple(i -> i <= ndims(mask) ? size(mask, i) : 1, max_ndim) for mask in noColon
    ]

    # Step 2: final broadcasted shape
    final_shape = getFinalShape(indices, size(array))

    # Step 3: verify
    verifyFancyIndexing(padded_shapes, max_ndim)

    # Step 4: allocate output
    output = NDArray{DType}(final_shape)

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

function fancyIndexing(array::NDArray{DType, N}, masks::NTuple{N, Union{Vector{Int}, Colon}} ) where {DType, N, T}

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
    output = NDArray{DType}(final_shape)

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


function fancyIndexing(array::NDArray{DType, N}, indices::NTuple{N, Union{AbstractArray{T}, Colon}} ) where {DType, N, T}

    # Verify that they have the same number of elements
    shapesMatch(array, indices) || throw(DIMENSIONS_DO_NOT_MATCH)

    # Create the array if the boolean was true
    filteredArray = [ val for (i, val) in enumerate(array) if elementIsTaken(mask[i]) ]

    # Return a new NDArray
    return NDArray{DType}(filteredArray)
    
end


function elementIsTaken(element::Union{AbstractArray{Bool}, Colon})
    return isa(element, Colon) || element == true
end


function shapesMatch(ndarray::NDArray{DType, N}, array::Vector{T}) where {DType, N, T}
    return ndarray.shape.length == length(array)
end

function shapesMatch(ndarray::NDArray{DType, N}, indices::NTuple{N, Union{Vector{Bool}, Colon}}) where {DType, N}
    shape = size(ndarray)
    return all( (index, dim) -> isa(index, Colon()) || length(index) == dim, zip(indices, shape))
end

function shapesMatch(ndarray::NDArray{DType, N}, indices::NTuple{N, Union{Vector{Int}, Colon}}) where {DType, N}
    shape = size(ndarray)
    return all( (index, dim) -> isa(index, Colon()) || length(index) <= dim, zip(indices, shape))
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









# Mixed Access: 
# ----------------

function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Indices} ) where {DType, N}

end


# ndarray[1, :, 3]
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Union{Int, Colon}} ) where {DType, N} end

# using cartesian index
function Base.getindex(A::NDArray, I::CartesianIndex) end

# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{AbstractArray{Bool}, 1}) where {DType, N} end

function findFinalShape(shape::Tuple, args::Vararg{Union{AbstractArray{Bool}, Colon}}) end


# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{Union{AbstractArray{Bool}, Colon}}) where {DType, N} end

# Fall back
Base.getindex(::NDArray{DType, N}, args) where {DType, N} = throw(INDICE_NOT_ACCEPTED)