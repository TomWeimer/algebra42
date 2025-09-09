mutable struct NDArray{DType, N} <: AbstractArray{DType, N}
   content::Union{Array{DType, 1}, Array{DType, 0}}
   shape::Shape{N}
   strides::Tuple{Vararg{Int}}
   
   # Constructor for generic data
   function NDArray{DType, 0}(data::Number) where {DType}
      shape = Shape(data)
      strides = _compute_strides(shape)
      content = fill(data)
      new{DType, 0}(content, shape, strides)
   end

   # Constructor for generic data
   function NDArray{DType}(data::Collection) where {DType}
      shape = Shape(data)
      strides = _compute_strides(shape)
      content = Array{DType, 1}(undef, shape.length);
      _fill!(data, shape, content, strides)
      new{DType, ndims(shape)}(content, shape, strides)
   end

   # Constructor for generic data
   function NDArray{Any}(data::Collection)
      shape = Shape(data, dtype=Any)
      strides = _compute_strides(shape)
      content = Array{Any, 1}(undef, shape.length);
      _fill!(data, shape, content, strides)
      new{Any, ndims(shape)}(content, shape, strides)
   end

    # Constructor for generic data
   function NDArray{DType, N}(content::Array{DType, N}, shape::Shape{N}, strides::Tuple{Vararg{Int}}) where {DType, N}
      new{DType, N}(content, shape, strides)
   end

   # Constructor for shape
    NDArray(shape::Shape; DType::Type = Float32) = new{DType, ndims(shape)}(Array{DType}(undef, shape.length), shape, _compute_strides(shape));

   # Constructor for shape
   NDArray{DType}(shape::Shape) where {DType} = new{DType, ndims(shape)}(Array{DType}(undef, shape.length), shape, _compute_strides(shape));
end

# Constructors:
# -------------------

# Constructor for scalar
function NDArray(scalar::Number; DType=typeof(scalar))
    return NDArray{DType, 0}(scalar);
end

function NDArray{DType}(scalar::Number) where {DType}
    return NDArray{DType, 0}(scalar)
end

# Helper: recursively index into nested vectors
function get_nested(data, I::CartesianIndex)
    x = data
    for i in Tuple(I)
        x = x[i]
    end
    return x
end

# Fill the preallocated array with converted values
function _fill!(data::Collection{Number}, shape::Shape, content::Array{DType}, strides::Tuple{Vararg{Int}}) where {DType}
   for i in ndims(shape)
        content[i] = convert(DType, data[i])
    end
    return content
end


# Fill the preallocated array with converted values
function _fill!(data::Collection{U}, shape::Shape{N}, content::Array{Any}, strides::Tuple{Vararg{Int}}) where {U, N}
  # @infiltrate
 
   for idx in CartesianIndices(shape.dims)
      content[_offset(strides, shape, idx)] = get_nested(data, idx)
    end
    return content
end

# Fill the preallocated array with converted values
function _fill!(data::Collection{U}, shape::Shape{N}, content::Array{DType}, strides::Tuple{Vararg{Int}}) where {DType, U, N}
  # @infiltrate
   for idx in CartesianIndices(shape.dims)
        content[_offset(strides, shape, idx)] = convert(DType, get_nested(data, idx))
    end
    return content
end


function _compute_strides(shape::Shape)
    dim = length(shape.dims)
    # Pre-allocate a tuple of zeros
    strds = zeros(Int, dim)
    # The stride for the first dimension is always 1
    strds[1] = 1
    # For subsequent dimensions, stride[i] = stride[i-1] * size[i-1]
    for i in 2:dim
        strds[i] = strds[i-1] * shape.dims[i-1]
    end
    return Tuple(strds)
end


# Getter:
# -------------------

function Base.length( array::NDArray )
   return array.shape.length;
end

function Base.size( array::NDArray )
   return array.shape.dims;
end

function Base.ndims( array::NDArray )
   return ndims(array.shape);
end


# Item
# -------------

# Extend Base.getproperty so arrays accept `.item`
function Base.getproperty(a::NDArray, s::Symbol)
    if s === :item
        return (index::Vararg{Int}) -> item(a, index)   # return a callable function
    else
        return getfield(a, s)  # default behavior
    end
end

"""
    item(array::NDArray{DType, 0}) -> DType

Return the scalar value stored in a 0-dimensional `NDArray` (a scalar array).

# Arguments
- `array::NDArray{DType, 0}`: A 0-dimensional NDArray containing a single value.
- `index::Vararg{Int}`: Optional. Must be empty; any indices will throw an error.

# Returns
- `DType`: The scalar value contained in the array.

# Throws
- `DomainError` if any index is provided, because a scalar cannot be indexed.
"""
function item( array::NDArray{DType, 0}, index::Vararg{Int})::DType where { DType }
   isempty(index) || throw(DomainError("Scalar NDArray does not accept indices"))
   return array.content[];
end



"""
    item(array::NDArray{DType}, idx::Vararg{Int, 1}) -> DType

Return an element from an N-dimensional `NDArray` given its index/indices.

# Arguments
- `array::NDArray{DType}`: An N-dimensional NDArray.
- `idx::Vararg{Int, 1}`: One or more indices specifying the element to return.
  If no index is provided, defaults to `1`.

# Returns
- `DType`: The element at the given index.

# Throws
- `DomainError` if the index is out of bounds.
"""
function item( array::NDArray{DType}, idx::Vararg{Int, 1}) where { DType }
   if (isempty(idx))
      index = 1
   else
      index = idx[1]
      _isValidIndex(array.shape, index) || throw(DomainError("The index is out of bounds"))
   end
   return array.content[index];
end

# Access
# -------------

# ndarray[()]

# works only in scalar
function Base.getindex(array::NDArray{DType, 0}, args::Tuple{}) where {DType}
   isempty(args) || throw(DomainError("Not working on scalar"))
   return item(array);
end

function Base.getindex(::NDArray{DType, N}, args::Tuple{Int}) where {DType, N}
   # Check that the tuple is not empty
   isempty(args) &&  throw(DomainError("The dimension is not 0 ndarray[()] works only on scalar"))
   # Throw an error the element can not be accessed with a tuple
   throw(DomainError("The element cannnot be accessed with tuple"))
end

# ndarray[1] not working in scalar
function Base.getindex(array::NDArray{DType, 0}, indices::Vararg{Union{Int, Colon}} ) where {DType}
   throw(DomainError("accessing by index with a scalar is only possible as ndarray[()] or ndarray.item()"))
end

# ndarray[1, 2, 3]
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Int} ) where {DType, N}
   #@infiltrate
   # Check that the indices are not empty
   isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
   # Check if the index are valid

   dim = ndims(array);
   indicesNb = length(indices)

   (indicesNb <= dim) ||  throw(DomainError("Indexes out of bounds"))
   
   indicesStretched = indices


   (elementsAreValid(size(array), indices...)) || throw(DomainError("Indexes out of bounds"))

   if (indicesNb < dim)
      indicesStretched = ntuple(i -> i <= indicesNb ? indices[i] : Colon(), dim)
   end

   # If the index is valid return the associated elements
   return array.content[_offset(array, indicesStretched...)];
end

# ndarray[1, :, 3]
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Union{Int, Colon}} ) where {DType, N}
   # Check that the indices are not empty
   isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
   # Check if the index are valid
   #@infiltrate
   _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
   # If the index is valid return the associated elements
   return @view array[indices...];
end

# using cartesian index
function Base.getindex(A::NDArray, I::CartesianIndex)
   A.content[_offset(A, I)]
end

# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{AbstractArray{Bool}, 1}) where {DType, N}

   # We receive a single 1 dimensional array, and we must return a 1 dimension array
   mask = args[1] 

   # The number of elements in the mask should match the number of elements in the array
   numberOfElementsMatch(array, mask) || throw( DomainError("The number of elements in the mask and the array should be equal") )

   # We then filter the elements based on the mask
   filteredArray = Vector{DType}()

   for (i, val) in enumerate(array)
      mask[i] && push!(filteredArray, val)
   end

   # And return a new ndarray with the filtered content
   out = NDArray{DType}(filteredArray)
   return out
end

function findFinalShape(shape::Tuple, args::Vararg{Union{AbstractArray{Bool}, Colon}})
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


# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{Union{AbstractArray{Bool}, Colon}}) where {DType, N}

   # We receive a single 1 dimensional array, and we must return a 1 dimension array
   mask = args[1] 

   shape = size(array)

   # The number of elements in the mask should match the number of elements in the array
   for (i, val) in enumerate(args)
      (isa(val, Colon) || numberOfElementsMatch(shape[i], val)) || throw( DomainError("The number dimensions of the mask should match") )
   end

    max_ndim = maximum(ndims.(masks))
    padded_shapes = [
        ntuple(i -> i <= ndims(mask) ? size(mask, i) : 1, max_ndim)
        for mask in masks
    ]

   # Step 2: final broadcasted shape
   final_shape = findFinalShape(shape, args)
   
   # Step 5: loop
    for idx in CartesianIndices(final_shape)
        # Compute the projected indices for each mask
        projected_indices = [project_index_function(idx, padded_shapes[j]) for j in 1:length(masks)]

        # Look up the actual integer indices from the masks
        real_indices = map(j -> masks[j][projected_indices[j]], 1:length(masks))

        # Use splatting to index into your NDArray
        output[idx] = array[real_indices...]
    end


   # We then filter the elements based on the mask
   filteredArray = Vector{DType}()

   for (i, val) in enumerate(array)
      mask[i] && push!(filteredArray, val)
   end

   # And return a new ndarray with the filtered content
   out = NDArray{DType}(filteredArray)
   return out
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

# Fancy indexing
function Base.getindex(array::NDArray{DType, N}, masks::Vararg{AbstractArray}) where {DType, N}
    isempty(masks) && throw(DomainError("Cannot access with ndarray[[]]"))

    # Step 1: pad shapes
    max_ndim = maximum(ndims.(masks))
    
    padded_shapes = [
        ntuple(i -> i <= ndims(mask) ? size(mask, i) : 1, max_ndim)
        for mask in masks
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

function numberOfElementsMatch(array::NDArray{DType, N}, mask::AbstractArray{Bool}) where {DType, N}
   return array.shape.length == length(mask)
end

function numberOfElementsMatch(dim, mask::AbstractArray{Bool})
   return dim == length(mask)
end





# Slicing



# ndarray[1] = 10
function Base.setindex!(a::NDArray{DType, N}, val, index::Int) where {DType, N}
   _isValidIndex(a.shape, index) || throw(DomainError("Invalid index when a[i] = ..."))
   convertedValue::DType = convert(DType, val)
   a.content[index] = convertedValue
end

# ndarray[1, 2, 3] = 10
function Base.setindex!(a::NDArray{DType, 0}, val, indices::Vararg{Int}) where {DType}
   # Check that the indices are empty
   isempty(indices) || throw(DomainError("Can access only a[()] if it is a scalar"))
   # Check if the index are valid
   convertedValue::DType = convert(DType, val)
   a.content[1] = convertedValue;
end


# ndarray[cartesianIndex] = 10
function Base.setindex!(a::NDArray{DType, N}, val, index::CartesianIndex) where {DType, N}
   idx = Tuple(index);
   _isValidIndex(a.shape, idx...) || throw(DomainError("Invalid index when a[i] = ..."))
   convertedValue::DType = convert(DType, val)
   a.content[_offset(a, idx...)] = convertedValue
end

function Base.setindex!(array::NDArray{DType, N}, val, indices::Vararg{Int}) where {DType, N}
   # Check that the indices are not empty
   isempty(indices) && throw(DomainError("Cannot access to element with ndarray[()]"))
   # Check if the index are valid
   _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
   convertedValue::DType = convert(DType, val)
   array.content[_offset(array, indices...)] = convertedValue;
end



"""
    Base.view(array::NDArray, inds...)

Crée une vue `NDArray` qui partage la mémoire avec `array`, 
mais avec des dimensions et strides recalculés.

C'est un `NDArray` modifiable : toute modification dans la vue 
affecte aussi le tableau original.
"""
function Base.view(array::NDArray{DType,N}, inds...) where {DType,N}
   # The key is to first get the linear indices that correspond to the multidimensional slice.
   linear_indices = LinearIndices(array.shape.dims)[inds...]

   return view(array.content, linear_indices)
end

# Iteration
# -------------

# for x in ndarray

# Define iteration
function Base.iterate(::NDArray{DType, 0}) where {DType}
   throw(DomainError("Cannot iterate on a scalar"))
end

function Base.iterate(ndarray::NDArray{DType, N}) where {DType, N}
   isempty(ndarray.content) && return nothing
   return ndarray.content[1], 2
end

function Base.iterate(ndarray::NDArray{DType, N}, state::Int) where {DType, N}
   state > length(ndarray.content) && return nothing
   return ndarray.content[state], state + 1
end

# Internal functions:
# -------------------

# _isValidIndex

# Check that the index is valid (flat index)
function _isValidIndex(shape::Shape, index::Int)
   return index >= 1 && index <= shape.length
end

# Check that the indices are valid (n indices)
function _isValidIndex(shape::Shape, indices::Vararg{Union{Int, Colon}})
   return dimensionMatch(shape, indices...) && elementsAreValid(shape.dims, indices...)
end

function dimensionMatch(shape::Shape, indices::Vararg{Union{Int, Colon}})
   return length(indices) == ndims(shape)
end

function dimensionMatch(shape::Shape, mask::Vararg{AbstractArray{Bool, 1}, 1})
   return length(mask) == shape.length # the mask has the same number of element than the ndarray
end

function dimensionMatch(shape::Shape, masks::Vararg{AbstractArray{Bool, 1}, N}) where {N}

   
   return length(mask) == ndims(shape)
end

function elementsAreValid(dims::Tuple, indices::Vararg{Int})
   return all((t) -> 1 <= t[1] <= t[2], zip(indices, dims))
end

function elementsAreValid(dims::Tuple, indices::Vararg{Union{Int, Colon}})
   return all((t) -> isa(t[1], Colon) || (1 <= t[1] <= t[2]), zip(indices, dims))
end

# _offset

# Obtain the index ( flat index ) from multiple indices
function _offset(ndarray::NDArray, indices::Vararg{Int})::Int
   offset = 0;
   for i in 1:ndims(ndarray)
      offset +=  (indices[i] - 1) * ndarray.strides[i];
   end
   return offset + 1;
end

# Obtain the index ( flat index ) from cartesian index
function _offset(ndarray::NDArray, I::CartesianIndex)::Int
   idx_tuple = Tuple(I)
   offset = 0;
   for i in 1:ndims(ndarray)
      offset += (idx_tuple[i] - 1) * ndarray.strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1;
end


# Obtain the index ( flat index ) from cartesian index
function _offset(strides::Tuple{Vararg{Int}}, shape::Shape{N}, I::CartesianIndex)::Int where {N}
   idx_tuple = Tuple(I)
   offset = 0;
   for i in 1:ndims(shape)
       offset += (idx_tuple[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1;
end

# 5. parent: Required for the SubArray to refer back to the original object
Base.parent(arr::NDArray) = arr