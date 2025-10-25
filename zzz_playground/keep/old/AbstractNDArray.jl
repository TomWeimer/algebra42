include("constant.jl")
include("macro.jl")
include("Errors.jl")

using Base: @propagate_inbounds

using .Errors: ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY

abstract type AbstractNDArray{T, N} <: AbstractArray{T, N} end

const ScalarNDArray{DType} = AbstractNDArray{DType,0}

const AllCartesianIndex{N} = Base.AbstractCartesianIndex{N}

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Functions to Overload:                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# To create a class that behave like a AbstractNDArray, you only need to overload the following functions:

# Return the indexStyle used by the array
IndexStyle(::Type{<:AbstractNDArray}) = @mustoverload

# Return the indexStyle used by the array
IsContingous(::Type{<:AbstractNDArray}) = @mustoverload

# Return the shape of the array
size(a::AbstractNDArray) = @mustoverload

# Return the parent array, or self if no parent exists
parent(a::AbstractNDArray) = @mustoverload

# Create a view of the element from indices
view(a::AbstractNDArray{DType,N}, inds...) where {DType,N} = @mustoverload


# Get and set elements:
# ---------------------

# Scalar
_getElement(a::ScalarNDArray) = @mustoverload
_setElement!(a::ScalarNDArray{T}, val::T) where T = @mustoverload

# LinearIndices
_getElement(a::ScalarNDArray, i::Int) = @mustoverload
_setElement!(a::ScalarNDArray{T}, val::T, i::Int) where T = @mustoverload

# CartesianIndex
_getElement(a::AbstractNDArray{DType, N}, I::CartesianIndex{N}) where {DType, N} = @mustoverload
_setElement!(a::AbstractNDArray{DType, N}, val::DType, I::CartesianIndex{N}) where {DType, N} = @mustoverload

# 'Normal' indices
_getElement(a::AbstractNDArray{T, N}, indices::NTuple{N, Any})          where {T, N} = @mustoverload
_setElement!(a::AbstractNDArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = @mustoverload


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                            Implementation of AbstractArray:                                   #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Simple implementation of axes
axes(a::AbstractNDArray) = map(Base.OneTo, size(a))

# Is needed when the array can be in an uninitialized state
isassigned(a::AbstractNDArray, i::Int) = checkbounds(Bool, a, i)

# Return a copy of the array
copy(a::AbstractNDArray) = copy!(similar(a), a)

copy!(dest::AbstractNDArray, src::AbstractNDArray) = (
    @boundscheck checkbounds(dest, src);
    @inbounds foreach(i -> dest[i] = src[i], eachindex(src, dest));
    return dest
)

# Handle the iteration of the arrays depending on their index style
iterate(a::AbstractNDArray) = _iterate_by_style(Base.IndexStyle(a), a)

# Return the first and last element
first(a::AbstractNDArray) = a[_first_index(IndexStyle(a), a)]
last(a::AbstractNDArray)  = a[ _last_index(IndexStyle(a), a)]


# Functions if their IndexStyle is IndexLinear :
_first_index(::IndexLinear, ::AbstractNDArray) = 1
_last_index(::IndexLinear, a::AbstractNDArray) = length(a)

_iterate_by_style(::IndexLinear, a::AbstractNDArray) = isempty(a) ? nothing : (a[1], 2)
_iterate_by_style(::IndexLinear, a::AbstractNDArray, state::Int) = state > length(a) ? nothing : (a[state], state + 1)


# Functions if their IndexStyle is IndexCartesian :

_first_index(::IndexCartesian, ::AbstractNDArray{T, N}) where {T, N} = CartesianIndex(ntuple(_ -> 1, N))
_last_index(::IndexCartesian, a::AbstractNDArray{T, N}) where {T, N} = CartesianIndex(ntuple(i -> size(a, i), N))

_iterate_by_style(::IndexCartesian, a::AbstractNDArray{T, N}) where {T, N} = (
    firstindex = _first_index(IndexCartesian, a);
    isempty(a) ? nothing : (a[firstindex], firstindex)
)

_iterate_by_style(::IndexCartesian, a::AbstractNDArray, state::CartesianIndex) = (
    next_index = inc(state, size(a));
    next_index === nothing ? nothing : (a[next_index], next_index)
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Getindex:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

getindex(a::ScalarNDArray, emptyTuple::Tuple{}) = _getElement(a, emptyTuple)

# Handle Linear index
getindex(a::AbstractNDArray{T, N}, i::Int) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _getElement(a, i)
)

# Handle Cartesian index
getindex(a::AbstractNDArray{T, N}, I::AllCartesianIndex{N}) where {T, N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _getElement(a, I)
)

# Handle multiple int index but not a cartesian index
getindex(a::AbstractNDArray{T,N}, inds::Vararg{Index, N}) where {T,N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _getElement(a, inds)
)

# Handle multiple int index (needing to be expanded) but not a cartesian index
getindex(a::AbstractNDArray{T, N}, inds::Vararg{Index}) where {T, N} = (
    @boundscheck checkbounds(a, inds...);
    @checknindices a inds;
    @inbounds a[ expand_indices(inds, N)... ]
)

# Handle index resulting in a view
getindex(a::AbstractNDArray{T,N}, inds::Vararg{ViewIndices, N}) where {T, N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds @view a[indices...]
)

# In simple cases, we know that we don't need to use axes(A), optimize those.
# Having this here avoids invalidations from multidimensional.jl: to_indices(A, I::Tuple{Vararg{Union{Integer, CartesianIndex}}})

    
# Handle indices resulting in a copy (fancy indexing)
getindex(a::AbstractNDArray{T,N}, inds::Vararg{CopyIndices}) where {T,N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds fancy_index(a, indices)
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Setindex:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

setindex!(a::ScalarNDArray{T}, val::T, ::Tuple{}) where {T} = _setElement!(a, val)

setindex!(a::AbstractNDArray{DType, N}, val::DType, inds::Vararg{Int, N}) where {DType, N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _setElement!(a, val, inds)
)

setindex!(a::AbstractNDArray{T,N}, val::T, I::CartesianIndex{N}) where {T,N} = (
    @boundscheck checkbounds(a,  I...);
    @inbounds _setElement!(a, val, I)
)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                    Get and Set index errors:                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

getindex(::ScalarNDArray,   inds::Int)            = @throw_index_error inds ERR_INDEX_INT_ON_SCALAR_ARRAY
getindex(::AbstractNDArray, inds::Tuple{})        = @throw_index_error inds ERR_EMPTY_TUPLE_ON_ND_ARRAY
getindex(::ScalarNDArray,   inds::ViewIndices...) = @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY


setindex!(::ScalarNDArray{T},     ::T, inds::Int)         where T     = @throw_index_error inds ERR_INDEX_INT_ON_SCALAR_ARRAY
setindex!(::AbstractNDArray{T,N}, ::T, inds::Tuple{})     where {T,N} = @throw_index_error inds ERR_EMPTY_TUPLE_ON_ND_ARRAY
setindex!(::ScalarNDArray{T}, ::T, inds::ViewIndices...)  where T     = @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                          Other functions not required by AbstractArray:                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Return the total number of elements of the array
length(a::AbstractNDArray) = prod(size(a))

# Return the type of the elements of the array
Base.eltype(::AbstractNDArray{T}) where {T} = T

# Return the number of dimension of the array
Base.ndims(::AbstractNDArray{T, N}) where {T, N} = N
