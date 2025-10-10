include("../../iterator/NestedArrayIndices.jl")



abstract type AbstractNDArray{DType, N} <: AbstractArray{DType,N} end

"""
   NDArray{DType,N}

Fields:
- `content`: Stores the array data as a 1D or 0D array.
- `shape`: Shape information for the array.
- `strides`: Tuple of strides for efficient indexing.

Provides constructors for scalars, ragged arrays, regular arrays, and matrices.
"""
mutable struct NDArray{DType,N} <: AbstractNDArray{DType,N}
   content::Union{Array{DType,1},Array{DType,0}}
   shape::Shape{N}
   strides::Tuple{Vararg{Int}}

   # Default constructor
   NDArray(content::AbstractArray{T}, shape::Shape{N}, strides) where {T,N} = new{T, N}(content, shape, strides)
end

# Constructors:
# -------------------

# scalar 
NDArray{T}(scalar::Number) where {T} = _init(scalar, T)

# ragged array
NDArray{Any}(data::Collection) = _init(data, Shape(data, dtype=Any), Any)

# array
NDArray{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
NDArray{DType}(data::AbstractArray{T, N}) where {DType, T, N} = _init(data, Shape(size(data)), DType)

# Uninitialized constructors from shape:

NDArray{DType}(shape::Tuple) where {DType} = NDArray{DType}(Shape(shape))

NDArray{DType}(shape::Shape{N}) where {DType, N} = NDArray( Array{DType}(undef, shape.length), shape, _compute_strides(shape) )


# Create a ndarray from range
function reshape(range::AbstractRange{T}, shape::NTuple{N})  where {T, N}
    length(range) == prod(shape) || throw(ArgumentError("Can't resize the range with the shape given"))
    ndarray = NDArray{T}(shape)
    for (i, val) in enumerate(range)
      ndarray[i] = val 
   end
   return ndarray
end

# Functions:
# ----------

"""
    flatten(array::NDArray{DType,N})

Returns the underlying content of the NDArray as a flat array.
"""
flatten(array::NDArray{DType,N}) where {DType,N} = array.content

"""
    item(a::NDArray, idx...)

Accesses an element of the NDArray using the provided indices.
For scalars, only empty indices are allowed.
"""
item(a::NDArray{DType,0}, idx::Vararg{Int}) where {DType} = isempty(idx) ? _getElement(a, 1) : throw(DomainError("Scalar NDArray does not accept indices"))

item(a::NDArray{DType}, idx::Vararg{Int,1}) where {DType} = isempty(idx) ? _getElement(a, 1) : _getElement(a, idx[1])



# Base functions (overloaded):
# ----------------------------


# Getter:

Base.size(array::NDArray) = array.shape.dims

Base.ndims(array::NDArray) = ndims(array.shape)

Base.length(array::NDArray) = array.shape.length

Base.eltype(A::NDArray{T}) where {T} = T

Base.pointer(A::NDArray{T}) where T = pointer(A.content)

# Setter:

Base.setindex!(a::NDArray{DType,0}, val::DType, ::Tuple{}) where {DType} = _setElement!(a, 1, val)

Base.setindex!(a::NDArray{DType}, val::DType, indices::Vararg{Int}) where {DType} = _setElement!(a, _getIndex(a, indices), val)

Base.setindex!(a::NDArray{DType,N}, val::DType, index::CartesianIndex) where {DType,N} = _setElement!(a, _getIndex(a, Tuple(index)), val)

# Iteration:

Base.iterate(::NDArray{DType,0}) where {DType} = throw(DomainError("Cannot iterate on a scalar"))

Base.iterate(ndarray::NDArray{DType,N}) where {DType,N} = isempty(ndarray.content) ? nothing : (ndarray.content[1], 2)

Base.iterate(ndarray::NDArray{DType,N}, state::Int) where {DType,N} = state > length(ndarray.content) ? nothing : (ndarray.content[state], state + 1)

# Similar:
function Base.similar(A::NDArray{T}) where {T}
    NDArray{T}(size(A))  # create a new NDArray of same size
end

function Base.similar(A::NDArray, ::Type{T}) where {T}
    NDArray{T}(size(A))
end

function Base.similar(A::NDArray{T}, dims::Tuple{Vararg{Int}}) where {T}
    NDArray{T}(dims)
end

# Copy:

function Base.copy(a::NDArray)
   dest = similar(a)
   copy!(dest, a)
   return dest
end

function Base.copy!(dest::NDArray, a::NDArray)
   for (i, val) in enumerate(a)
      dest[i] = val
   end
end

# View:

function Base.view(array::NDArray{DType,N}, inds...) where {DType,N}
   # The key is to first get the linear indices that correspond to the multidimensional slice.
   linear_indices = LinearIndices(array.shape.dims)[inds...]

   return view(array.content, linear_indices)
end

# Property:

function Base.getproperty(a::NDArray, s::Symbol)
   if s === :item
      return (index::Vararg{Int}) -> item(a, index)   # return a callable function
   else
      return getfield(a, s)  # default behavior
   end
end


# Parent:

Base.parent(arr::NDArray) = arr


# Internal functions:
# ------------------


"""
    _init(data, ...)

This internal function acts as a unified factory for creating `NDArray` instances from a
variety of starting data types. Using multiple dispatch, it funnels all inputs
(whether a scalar or an array) through a series of steps to ensure the final
object is correctly constructed with its content, shape, and strides.
"""
_init(scalar::Number, T::Type) = _init( fill( T(scalar) ), Shape(scalar) )

# array
_init(data::AbstractArray{U}, shape::Shape{N}, T::Type, strides=_compute_strides(shape))  where {U, N} = _init( _fill(data, shape, strides, T), shape, strides )

# all 
_init(content::AbstractArray{T}, shape::Shape{N}, strides=_compute_strides(shape)) where {T, N} = NDArray(content, shape, strides)


"""
    _compute_strides(shape::Shape)

Computes the strides for the given shape, used for efficient indexing.
"""
_compute_strides(shape::Shape) = _compute_strides(shape.dims)

function _compute_strides(dims::Tuple)
   dim = length(dims) == 0 ? 1 : length(dims)
   strds = zeros(Int, dim)
   # The stride for the first dimension is always 1
   strds[1] = 1
   # For subsequent dimensions, stride[i] = stride[i-1] * size[i-1]
   for i in 2:dim
      strds[i] = strds[i-1] * dims[i-1]
   end
   return Tuple(strds)
end


"""
    _fill(data, shape, strides, dtype)

Internal function to fill the NDArray content from the provided data.
Handles regular arrays, ragged arrays, and nested arrays.
"""
function _fill(data::AbstractArray{T,N}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {T,N}
   content = Array{dtype, 1}(undef, shape.length)
   
   for (i, val) in enumerate(data)
      content[i] = dtype(val)
   end
   
   return content
end

# Fill the content from nested array
_fill(d::Collection{U}, s::Shape{N}, st::Tuple{Vararg{Int}}, dt::Type) where {U, N} =
   (dt == Any) ? _fill_ragged_array(d, s,st, dt) : (N >= 3) ? _fill_from_nested_indices(d, s, st, dt) : _fill_from_offset(d, s, st, dt)

function _fill_ragged_array(data::Collection{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
   content = Array{dtype, 1}(undef, shape.length)

   for i in 1:shape.length
      content[i] = data[i]
   end
   return content
end

function _fill_from_offset(data::Collection{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
   content = Array{dtype, 1}(undef, shape.length)

   for idx in CartesianIndices( size(shape) )
      content[_offset(strides, shape, idx)] = dtype( _get_nested(data, idx) )
   end
   return content
end


function _fill_from_nested_indices(data::Collection{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
   content = Array{dtype, 1}(undef, shape.length)
   
   all_index = NestedArrayIndices( size(shape) )
   for (i, I) in enumerate(all_index)
      content[i] = dtype( _get_nested(data, I) )
   end
   return content
end

"""
    _get_nested(data, I::CartesianIndex)

Recursively indexes into nested vectors to retrieve the element at the given Cartesian index.
"""
function _get_nested(data, I::CartesianIndex)
   x = data
   for i in Tuple(I)
      x = x[i]
   end
   return x
end

"""
    _offset(ndarray::NDArray, indices...)

Computes the flat index in the underlying content array from multi-dimensional indices.
"""
function _offset(ndarray::NDArray, indices::Vararg{Int})::Int
   offset = 0
   for i in 1:ndims(ndarray)
      offset += (indices[i] - 1) * ndarray.strides[i]
   end
   return offset + 1
end

# Obtain the index ( flat index ) from cartesian index
function _offset(ndarray::NDArray, I::CartesianIndex)::Int
   idx_tuple = Tuple(I)
   offset = 0
   for i in 1:ndims(ndarray)
      offset += (idx_tuple[i] - 1) * ndarray.strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1
end


# Obtain the index ( flat index ) from cartesian index
function _offset(strides::Tuple{Vararg{Int}}, shape::Shape{N}, I::CartesianIndex)::Int where {N}
   idx_tuple = Tuple(I)
   offset = 0
   for i in 1:ndims(shape)
      offset += (idx_tuple[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1
end

"""
    _getIndex(array::NDArray, indices...)

Validates and computes the flat index for element access.
Throws an error for invalid indices.
"""
function _getIndex(array::NDArray, indices::Tuple{Vararg{Int}})
   index = _offset(array, indices...)
   (isempty(indices) && N != 0  ) && throw(DomainError("Can access only a[()] if it is a scalar"))
   _isValidIndex(array.shape, index) || throw(DomainError("Invalid index when a[i] = ..."))
   return index
end

function _getIndex(array::NDArray, indices::Tuple{Int})
  # println("array: ", array.content, " index obtained: ", index)
   _isValidIndex(array.shape, indices[1]) || throw(DomainError("Invalid index when a[i] = ..."))
   return indices[1]
end

function _getIndex(array::NDArray, index::Int)
  # println("array: ", array.content, " index obtained: ", index)
   _isValidIndex(array.shape, index) || throw(DomainError("Invalid index when a[i] = ..."))
   return index
end



"""
    _getElement(a::NDArray, index::Int)

Returns the element at the given flat index from the NDArray.
"""
function _getElement(a::NDArray{DType,N}, index::Int) where {DType,N}
   a.content[_getIndex(a, index)]
end

"""
    _setElement!(a::NDArray, index::Int, val)

Sets the element at the given flat index to the provided value.
"""
function _setElement!(a::NDArray{DType,N}, index::Int, val) where {DType,N}
   a.content[index] = DType(val)
end

"""
    _isValidIndex(shape::Shape, index::Int)

Checks if the flat index is within bounds for the given shape.
"""
_isValidIndex(shape::Shape, index::Int) = 1 <= index <= shape.length

_isValidIndex(shape::Shape, indices::Vararg{Union{AbstractRange, Int, Colon}}) = all((t) -> t[1] isa Colon || t[1] isa AbstractRange && 1 <= first(t[1]) <= last(t[1]) <= t[2] || 1 <= t[1] <= t[2], zip(indices, shape.dims))