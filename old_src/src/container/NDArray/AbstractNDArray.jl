using Base: @propagate_inbounds
import Base: *, +, -
using .Errors: ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY, ERR_IDX_TYPE_VECTOR

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       AbstractNDArray:                                        #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type AbstractNDArray{T, N} <: AbstractArray{T, N} end

const ScalarNDArray{T} = AbstractNDArray{T,0}

const AbstractNDMatrix{T} = AbstractNDArray{T, 2}

# ======== Functions to overload ============================================================== #

# Base functions:

Base.axes(a::AbstractNDArray)               = @mustoverload
Base.parent(a::AbstractNDArray)             = @mustoverload
Base.view(a::AbstractNDArray, inds...)      = @mustoverload
Base.IndexStyle(::Type{<:AbstractNDArray})  = @mustoverload

# Internal functions:

# Scalar
_getElement(::ScalarNDArray)        = @mustoverload
_setElement!(::ScalarNDArray, val)  = @mustoverload

# LinearIndices
_getElement(::AbstractNDArray,       i::Int) = @mustoverload
_setElement!(::AbstractNDArray, val, i::Int) = @mustoverload

# CartesianIndex
_getElement(::AbstractNDArray{T, N},       I::AllCartesianIndex{N}) where {T, N} = @mustoverload
_setElement!(::AbstractNDArray{T, N}, val, I::AllCartesianIndex{N}) where {T, N} = @mustoverload

# 'Normal' indices
_getElement(::AbstractNDArray{T, N},       indices::NTuple{N, Index}) where {T, N} = @mustoverload
_setElement!(::AbstractNDArray{T, N}, val, indices::NTuple{N, Index}) where {T, N} = @mustoverload


# ======== Functions for all AbstractNDArray ================================================== #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       core properties:                                        #
#_______________________________________________________________________________________________#

Base.eltype(::Type{ <: AbstractNDArray{T} })  where T = T
Base.eltype(a::AbstractNDArray{T})   where T = T

Base.ndims(::AbstractNDArray{T, N})     where {T, N} = N
Base.size(a::AbstractNDArray)                   = map(length, axes(a))
Base.length(a::AbstractNDArray)                 = prod(size(a))

# Scalar
Base.strides(::ScalarNDArray)                = ()
Base.length(a::ScalarNDArray)                = 1

function Base.vec(a::AbstractNDArray)
    v = similar(a, eltype(a), (length(a), ))
    for (i, val) in enumerate(a)
        v[i] = val
    end
    return v
end 


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       copy functions:                                         #
#_______________________________________________________________________________________________#

Base.copy(a::AbstractNDArray) = copy!(similar(a), a)

function Base.copy!(dest::ScalarNDArray, src::AbstractNDArray)
    @boundscheck size(dest) == size(src)
    dest[()] = src[()]
    dest
end

function Base.copy!(dest::AbstractNDArray, src::AbstractNDArray)
    @boundscheck size(dest) == size(src);
    for i in eachindex(src)
        dest[i] = src[i]
    end
    dest
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                            indexing:                                          #
#_______________________________________________________________________________________________#

Base.first(a::AbstractNDArray) = a[ firstindex(a) ]
Base.last(a::AbstractNDArray)  = a[ lastindex(a)  ]


Base.firstindex(a::AbstractNDArray) = first(eachindex(a))
Base.lastindex(a::AbstractNDArray)  = last(eachindex(a))

# eachindex, returns an iterable compatible with the array A

Base.eachindex(::IndexLinear, A::ScalarNDArray) = 1:1

Base.eachindex(::IndexLinear, A::AbstractNDArray{T, N}) where {T, N} = (
    isempty(A) && return (1:0);
    N == 1 ? axes(A, 1) : (1:length(A))
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       iteration:                                              #
#_______________________________________________________________________________________________#

Base.iterate(a::AbstractNDArray) = isempty(a) ? nothing : ( a[ firstindex(a) ], firstindex(a) )

Base.iterate(a::AbstractNDArray, state::Int) = (
    next_state = state + 1;
    next_state > length(a) ? nothing : (a[ next_state ], next_state)
)

Base.iterate(a::AbstractNDArray, state::AllCartesianIndex) = (
    next_index = inc(state, size(a));
    next_index === nothing ? nothing : (a[next_index], next_index)
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                         setters:                                              #
#_______________________________________________________________________________________________#

Base.setindex!(a::ScalarNDArray{T}, val::T, idx::Union{Tuple{}, Nothing} = nothing) where T = (
    ( isnothing(idx) || idx == () ) ? _getElement(a) : @throw_index_error idx ERR_INDEX_INT_ON_SCALAR_ARRAY
)

Base.setindex!(a::AbstractNDArray{T,N}, val::T,  i::Int) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _setElement!(a, val, i)
)

Base.setindex!(a::AbstractNDArray{T,N}, val::T, I::Base.AbstractCartesianIndex{N}) where {T,N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _setElement!(a, val, I)
)

Base.setindex!(a::AbstractNDArray{DType, N}, val::DType, inds::Vararg{Int, N}) where {DType, N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _setElement!(a, val, inds)
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                         getters:                                              #
#_______________________________________________________________________________________________#

Base.getindex(a::ScalarNDArray{T}, idx::Union{Tuple{}, Nothing} = nothing) where T = (
    ( isnothing(idx) || idx == () ) ? _getElement(a) : @throw_index_error idx ERR_INDEX_INT_ON_SCALAR_ARRAY
)

Base.getindex(a::AbstractNDArray{T, N}, i::Int) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _getElement(a, i)
)

Base.getindex(a::AbstractNDArray{T, N}, I::AllCartesianIndex{N}) where {T, N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _getElement(a, I)
)

Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{Index}) where {T,N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _getElement(a, inds)
)

# create a view
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{ViewIndices}) where {T, N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds @view a[indices...]
)

# create a copy
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{CopyIndices}) where {T,N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds fancy_index(a, indices)
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                getters/setters errors:                                        #
#_______________________________________________________________________________________________#

Base.getindex(::AbstractNDArray{T, 1},   inds::NTuple{1}) where T  = @throw_index_error inds ERR_IDX_TYPE_VECTOR
Base.setindex!(::AbstractNDArray{T, 1},  inds::NTuple{1}) where T  = @throw_index_error inds ERR_IDX_TYPE_VECTOR

Base.setindex!(::ScalarNDArray, val, inds::ViewIndices...)         = @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY


# ======== Math: operator overload ================================================== #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                      broadcast overload:                                      #
#_______________________________________________________________________________________________#

# permits to use the broadcast syntax suxh as v .+= λ .* v

# broadcast style
struct NDArrayStyle2 <: Base.BroadcastStyle end

# when to use the broadcast style
Base.Broadcast.BroadcastStyle(::Type{<:AbstractNDArray}) = NDArrayStyle2()
Base.BroadcastStyle(::NDArrayStyle2, ::NDArrayStyle2) = NDArrayStyle2()
Base.BroadcastStyle(::NDArrayStyle2, ::Base.Broadcast.DefaultArrayStyle) = NDArrayStyle2()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle, ::NDArrayStyle2) = NDArrayStyle2()

# This defines how Julia should allocate the output array when performing a broadcast operation involving AbstractNDArray type.
Base.similar(bc::Base.Broadcast.Broadcasted{NDArrayStyle2}, ::Type{ElType}) where {ElType} = begin
    axes_bc = Base.Broadcast.axes(bc)
    dims = tuple(length.(axes_bc)...)

    dest = NDArray{ElType}(dims)
    return dest
end

# How to actually fill it with the broadcasted values
function Base.Broadcast.copy!(dest::AbstractNDArray, bc::Broadcast.Broadcasted{NDArrayStyle2})
    for (idx, val) in zip(eachindex(dest), bc)
        dest[idx] = val
    end
    return dest
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                      operators overload:                                      #
#_______________________________________________________________________________________________#

# --- Addition ---
+(a::AbstractNDArray, b::AbstractNDArray) = add(a, b)
+(a::Number, b::AbstractNDArray) = +(b, a)
+(a::AbstractNDArray, b::Number) = add(a, b)

# --- Subtraction ---
-(a::AbstractNDArray, b::AbstractNDArray) = sub(a, b)
-(a::Number, b::AbstractNDArray) = -(b, a)
-(a::AbstractNDArray, b::Number) = sub(a, b)

# --- Scalar Multiplication ---
*(a::AbstractNDArray, b::Number) = prod(a, b)
*(a::Number, b::AbstractNDArray) =  *(b, a)