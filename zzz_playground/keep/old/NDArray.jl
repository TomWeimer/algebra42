include("../../iterator/NestedArrayIndices.jl")
include("../Shape.jl")

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

const NestedArray{T} = AbstractArray{T, 1} # can also be single dimension array

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                      Constructors:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
NDArray{T}(x::Number) where {T} = (
   _init(x, T)
)

# Regular nested array
NDArray{T}(data::NestedArray{T}) where {T} = (
    _init(data, Shape(data), T)
)

# Ragged nested array
NDArray{Any}(data::NestedArray{T}) where {T} = (
    _init(data, Shape(data; dtype=Any), Any)
)

# From multidimensional array
NDArray{T}(data::AbstractArray{U, N}) where {T, U, N} = (
    _init(data, Shape(size(data)), T)
)

# From shape (tuple)
NDArray{T}(shape::Tuple) where {T} = (
    NDArray{T}(Shape(shape))
)

# From shape object
NDArray{T}(shape::Shape{N}) where {T, N} = (
    NDArray(Array{T}(undef, shape.length), shape, compute_strides(shape))
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                          _init:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
_init(x::Number, T::Type) = (
    _init(fill(T(x)), Shape(x))
)

# Array with explicit type and shape
_init(data::AbstractArray{U}, shape::Shape{N}, T::Type, strides=compute_strides(shape)) where {U, N} = (
    _init(_fill(data, shape, strides, T), shape, strides)
)

# Generic catch-all
_init(content::AbstractArray{T}, shape::Shape{N}, strides=compute_strides(shape)) where {T, N} = (
    NDArray(content, shape, strides)
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         _fill:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Fill using multidimentional array
function _fill(data::AbstractArray{T,N}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {T,N}
   content = Array{dtype, 1}(undef, shape.length)
   for (i, val) in enumerate(data)
      content[i] = dtype(val)
   end
   return content
end

function _fill(data::NestedArray{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    if dtype === Any
        return _fill_ragged_array(data, shape, strides, dtype)
    end
    return N ≥ 3 ? _fill_from_nested_indices(data, shape, strides, dtype) :  _fill_from_offset(data, shape, strides, dtype)
end

# Fill ragged array
function _fill_ragged_array(data::NestedArray{U}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    content = similar(data, dtype, shape.length)
    for i in eachindex(content)
        content[i] = data[i]
    end
    content
end

# Fill dim 1 and dim 2
function _fill_from_offset(data::NestedArray{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    content = similar(data, dtype, shape.length)
    for I in CartesianIndices(size(shape))
        content[_offset(strides, Tuple(I))] = dtype(_get_nested(data, I))
    end
    content
end

# Fill dim 3+
function _fill_from_nested_indices(data::AbstractArray{U, 1}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
   content = similar(data, dtype, shape.length)
   for (i, I) in enumerate( NestedArrayIndices( size(shape) ) )
      content[i] = dtype( _get_nested(data, I) )
   end
   return content
end

_get_nested(data, I::CartesianIndex) = foldl(getindex, Tuple(I); init=data)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Similar:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Similar: (create array of same size but uninitialized)
Base.similar(A::NDArray{T}) where {T} = NDArray{T}(size(A))

Base.similar(A::NDArray, ::Type{T}) where {T} = NDArray{T}(size(A))

Base.similar(::NDArray{T}, dims::Tuple{Vararg{Int}}) where {T} = NDArray{T}(dims)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Other functions :                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Parent:
flatten(array::NDArray{DType,N}) where {DType,N} = array.content

Base.pointer(A::NDArray{T}) where T = pointer(A.content)

# Create a ndarray from range
function reshape(range::AbstractRange{T}, shape::NTuple{N})  where {T, N}
    length(range) == prod(shape) || throw(ArgumentError("Can't resize the range with the shape given"))
    ndarray = NDArray{T}(shape)
    for (i, val) in enumerate(range)
      ndarray[i] = val 
   end
   return ndarray
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Implementation AbstractNDArray:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

IndexStyle(::Type{NDArray}) = IndexLinear()

IsContingous(::Type{NDArray}) = Val(true)

size(a::NDArray)   = a.shape.dims

parent(a::NDArray) = a

view(a::NDArray, inds...) = NDArrayView(a, inds)


# Get and set elements:
# ---------------------

# Scalar:
# -------

@propagate_inbounds _getElement(a::NDArray{T, 0}) where {T} = (
   a.content[]
)

@propagate_inbounds _setElement!(a::NDArray{T, 0}, val::T) where {T} = (
   a.content[] = val
)

# Linear index:
# ------------

@propagate_inbounds _getElement(a::NDArray{T, N}, i::Int) where {T, N} = (
   a.content[i]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, i::Int) where {T, N} = (
   a.content[i] = val
)

# Cartesian index:
# ---------------

@propagate_inbounds _getElement(a::NDArray{T, N}, I::CartesianIndex{N}) where {T, N} = (
   a.content[ offset(a, I) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, I::CartesianIndex{N}) where {T, N} = (
   a.content[ offset(a, I) ] = val
)

# 'Normal' index:
# ---------------

@propagate_inbounds _getElement(a::NDArray{T, N}, indices::NTuple{N, Any}) where {T, N} = ( 
   a.content[ offset(a, indices) ] 
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
   a.content[ offset(a, indices) ] = val
)















































# Internal functions:
# ------------------

# """
#     _getElement(a::NDArray, index::Int)

# Returns the element at the given flat index from the NDArray.
# """
# function _getElement(a::NDArray{DType,N}, index::Int) where {DType,N}
#    a.content[_getIndex(a, index)]
# end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         _setElement:                                          #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# """
#     _setElement!(a::NDArray, index::Int, val)

# Sets the element at the given flat index to the provided value.
# """
# function _setElement!(a::NDArray{DType,N}, index::Int, val) where {DType,N}
#    a.content[index] = DType(val)
# end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         _isValidIndex:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# """
#     _isValidIndex(shape::Shape, index::Int)

# Checks if the flat index is within bounds for the given shape.
# """
# _isValidIndex(shape::Shape, index::Int) = 1 <= index <= shape.length

# _isValidIndex(shape::Shape, indices::Vararg{Union{AbstractRange, Int, Colon}}) = all((t) -> t[1] isa Colon || t[1] isa AbstractRange && 1 <= first(t[1]) <= last(t[1]) <= t[2] || 1 <= t[1] <= t[2], zip(indices, shape.dims))




# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         Constants:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# # Return a single elements of the array ( if enough are provided )
# const ElementIndex = Int                          

# # Return a view if:
# # - a colon is present
# # - the number of indices given is less than the number of array's dimensions
# const ViewIndices = Union{AbstractRange, Int, Colon} 

# # Fancy indices are arrays of integer or boolean
# const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# # Return a copy of the array, if at least one fancy index is present
# const CopyIndices = Union{AbstractRange, Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         Get Index:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# function Base.getindex(array::NDArray{DType,0}, ::Tuple{}) where {DType}
#     return item(array)
# end


# function Base.getindex(::NDArray{DType,0}, indices::Int) where {DType}  # return an error
#     throw(DomainError("accessing by index with a scalar is only possible as ndarray[()] or ndarray.item()"))
# end

# function Base.getindex(::NDArray{DType,N}, args::Tuple{}) where {DType,N} # return an error
#     # Check that the tuple is not empty
#     isempty(args) && throw(DomainError("The dimension is not 0 ndarray[()] works only on scalar"))
#     # Throw an error the element can not be accessed with a tuple
#     throw(DomainError("The element cannnot be accessed with tuple"))
# end


# Base.getindex(::NDArray{DType,0}, ::Vararg{ViewIndices}) where {DType} = throw(DomainError("Cannot access to scalar with ndarray[:]"))



# Accessing N-dimensional array:
# -----------------------------

# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element

#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

#     # Check if the index are valid
#     (elementsAreValid(size(array), indices...)) || throw(DomainError("Indexes out of bounds"))

#     # If the index is valid return the associated elements
#     return array.content[_offset(array, indices...)]
# end


# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

#     # Check if the number of index is valid
#     dimIndices, dimArray = length(indices), ndims(array)

#     (dimIndices <= dimArray) || throw(DomainError("The number of index provided is too big"))

#     indicesStretched = (dimIndices == dimArray) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimArray)

#     # If the index is valid return the associated elements
#     return array[indicesStretched...]
# end


# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ViewIndices}) where {DType,N}
#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
#     # Check if the index are valid
#     #@infiltrate
#     _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
#     # If the index is valid return the associated elements
#     return @view array[indices...]
# end


# special indices:
# ----------------


# function Base.getindex(A::NDArray{T, N}, I::CartesianIndex{N}) where {T, N}
#     A.content[_offset(A, I)]
# end

# fancy indices:
# ----------------

# function Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}
#     return fancy_index(array, indices)
# end



# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{CopyIndices}) where {DType,N}
#    # println("enter motherfucker2")
#     return fancy_index(array, indices)
# end


# function elementsAreValid(dims::Tuple, indices::Vararg{Int})
#    return all((t) -> 1 <= t[1] <= t[2], zip(indices, dims))
# end

# function elementsAreValid(dims::Tuple, indices::Vararg{Union{Int,Colon}})
#    return all((t) -> isa(t[1], Colon) || (1 <= t[1] <= t[2]), zip(indices, dims))
# end




# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Set index:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Base.setindex!(a::NDArray{DType,0}, val::DType, ::Tuple{}) where {DType} = _setElement!(a, 1, val)

# Base.setindex!(a::NDArray{DType}, val::DType, indices::Vararg{Int}) where {DType} = _setElement!(a, _getIndex(a, indices), val)

# Base.setindex!(a::NDArray{DType,N}, val::DType, index::CartesianIndex) where {DType,N} = _setElement!(a, _getIndex(a, Tuple(index)), val)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Iteration:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Base.iterate(::NDArray{DType,0}) where {DType} = throw(DomainError("Cannot iterate on a scalar"))

# Base.iterate(ndarray::NDArray{DType,N}) where {DType,N} = isempty(ndarray.content) ? nothing : (ndarray.content[1], 2)

# Base.iterate(ndarray::NDArray{DType,N}, state::Int) where {DType,N} = state > length(ndarray.content) ? nothing : (ndarray.content[state], state + 1)



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                          View:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# function Base.view(array::NDArray{DType,N}, inds...) where {DType,N}
#    # The key is to first get the linear indices that correspond to the multidimensional slice.
#    linear_indices = LinearIndices(array.shape.dims)[inds...]

#    return view(array.content, linear_indices)
# end