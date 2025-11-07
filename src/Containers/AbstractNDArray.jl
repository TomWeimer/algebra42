import Base: *, +, -

using .Errors: ERR_INDEX_INT_ON_SCALAR_ARRAY, 
               ERR_EMPTY_TUPLE_ON_ND_ARRAY, 
               ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY, 
               ERR_IDX_TYPE_VECTOR
               
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                         AbstractNDArray:                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type AbstractNDArray{T, N} <: AbstractArray{T, N} end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #

const AbstractScalarNDArray{T}    = AbstractNDArray{T,0}
const AbstractNDMatrix{T} = AbstractNDArray{T,2}

# ════════════════════════════════════ functions to overload ═════════════════════════════════════ #
        
# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.axes(a::AbstractNDArray)               = @mustoverload
Base.parent(a::AbstractNDArray)             = @mustoverload
Base.IndexStyle(::Type{<:AbstractNDArray})  = @mustoverload

# ──── Internal functions ──────────────────────────────────────────────────────────────────────── #
            
_getElement(::AbstractScalarNDArray)        = @mustoverload
_setElement!(::AbstractScalarNDArray, val)  = @mustoverload

_getElement(::AbstractNDArray,       i::Index) = @mustoverload
_setElement!(::AbstractNDArray, val, i::Index) = @mustoverload

# TODO: check if we can merge them
_getElement(::AbstractNDArray{T, N},       I::AllCartesianIndex{N}) where {T, N} = @mustoverload
_setElement!(::AbstractNDArray{T, N}, val, I::AllCartesianIndex{N}) where {T, N} = @mustoverload

_getElement(::AbstractNDArray{T, N},       indices::NTuple{N, Index}) where {T, N} = @mustoverload
_setElement!(::AbstractNDArray{T, N}, val, indices::NTuple{N, Index}) where {T, N} = @mustoverload


# ══════════════════════════════ functions for all AbstractNDArray ═══════════════════════════════ #

# ──── core properties ─────────────────────────────────────────────────────────────────────────── #

Base.size(a::AbstractNDArray) = map(length, axes(a))
Base.length(a::AbstractNDArray) = prod(size(a))
Base.ndims(::AbstractNDArray{T, N})     where {T, N} = N

# TODO: check that we can remove this
Base.eltype(::Type{ <: AbstractNDArray{T} }) where T = T
Base.eltype(a::AbstractNDArray{T})           where T = T

# ──── copy functions ──────────────────────────────────────────────────────────────────────────── #
            
Base.copy(a::AbstractNDArray) = copy!(similar(a), a)

function Base.copy!(dest::AbstractNDArray, src::AbstractNDArray)
    @boundscheck size(dest) == size(src);
    for i in eachindex(src)
        dest[i] = src[i]
    end
    dest
end

# ──── indexing ────────────────────────────────────────────────────────────────────────────────── #

Base.first(a::AbstractNDArray) = a[firstindex(a)]
Base.last(a::AbstractNDArray)  = a[lastindex(a)]

Base.firstindex(a::AbstractNDArray) = first(eachindex(a))
Base.lastindex(a::AbstractNDArray)  = last(eachindex(a))

Base.eachindex(::IndexLinear, A::AbstractNDArray{T, N}) where {T, N} = (
    isempty(A) && return (1:0);
    N == 1 ? axes(A, 1) : (1:length(A))
)

# ──── iteration ───────────────────────────────────────────────────────────────────────────────── #

Base.iterate(a::AbstractNDArray) = isempty(a) ? nothing : ( a[ firstindex(a) ], firstindex(a) )

Base.iterate(a::AbstractNDArray, state::Int) = (
    next_state = state + 1;
    next_state > length(a) ? nothing : (a[ next_state ], next_state)
)

Base.iterate(a::AbstractNDArray, state::AllCartesianIndex) = (
    next_index = inc(state, size(a));
    next_index === nothing ? nothing : (a[next_index], next_index)
)

# ═══════════════════════════════════════════ indexing ═══════════════════════════════════════════ #
            
# ──── linear Index ────────────────────────────────────────────────────────────────────────────── #

Base.getindex(a::AbstractNDArray{T, N}, i::Index) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _getElement(a, i)
)

Base.setindex!(a::AbstractNDArray{T,N}, val,  i::Index) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _setElement!(a, val, i)
)

# ──── multiple Index ──────────────────────────────────────────────────────────────────────────── #

Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{Index}) where {T,N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _getElement(a, inds)
)

Base.setindex!(a::AbstractNDArray{T, N}, val, inds::Vararg{Index, N}) where {T, N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _setElement!(a, val, inds)
)

# ──── CartesianIndex ──────────────────────────────────────────────────────────────────────────── #

Base.getindex(a::AbstractNDArray{T, N}, I::AllCartesianIndex{N}) where {T, N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _getElement(a, I)
)

Base.setindex!(a::AbstractNDArray{T,N}, val, I::AllCartesianIndex{N}) where {T,N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _setElement!(a, val, I)
)

# ──── view and advanced indexing ──────────────────────────────────────────────────────────────── #

Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{ViewIndices}) where {T, N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds @view a[indices...]
)

Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{CopyIndices}) where {T,N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds fancy_index(a, indices)
)

# ──── indexing errors ─────────────────────────────────────────────────────────────────────────── #
            
Base.getindex(::AbstractNDArray{T, 1},   inds::NTuple{1}) where T = (
     @throw_index_error inds ERR_IDX_TYPE_VECTOR
)

Base.setindex!(::AbstractNDArray{T, 1},  inds::NTuple{1}) where T  = (
    @throw_index_error inds ERR_IDX_TYPE_VECTOR
)

# ═════════════════════════════════════════ scalar array ═════════════════════════════════════════ #
            
Base.strides(::AbstractScalarNDArray) = ()
Base.length(a::AbstractScalarNDArray) = 1

function Base.copy!(dest::AbstractScalarNDArray, src::AbstractNDArray)
    @boundscheck size(dest) == size(src)
    dest[()] = src[()]
    dest
end

Base.eachindex(::IndexLinear, A::AbstractScalarNDArray) = 1:1

Base.getindex(a::AbstractScalarNDArray{T}, idx::ZeroDimIndex = nothing) where T = (
    ( isnothing(idx) || idx == () ) ? _getElement(a) 
                                    : @throw_index_error idx ERR_INDEX_INT_ON_SCALAR_ARRAY
)

Base.setindex!(a::AbstractScalarNDArray, val, idx::ZeroDimIndex = nothing) = (
    ( isnothing(idx) || idx == () ) ? _setElement!(a, val) 
                                    : @throw_index_error idx ERR_INDEX_INT_ON_SCALAR_ARRAY
)

Base.setindex!(::AbstractScalarNDArray, val, inds::ViewIndices...)  = (
    @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY
)

# TODO: check if we can remove this
# ═══════════════════════════════════════ array arithmetic ═══════════════════════════════════════ #
            
# ──── addition ────────────────────────────────────────────────────────────────────────────────── #
            
+(a::AbstractNDArray, b::AbstractNDArray) = add(a, b)
+(a::AbstractNDArray, b::Number)          = add(a, b)
+(a::Number, b::AbstractNDArray)          = +(b, a)


# ──── subtraction ─────────────────────────────────────────────────────────────────────────────── #
            
-(a::AbstractNDArray, b::AbstractNDArray) = sub(a, b)
-(a::AbstractNDArray, b::Number)          = sub(a, b)
-(a::Number, b::AbstractNDArray)          = -(b, a)


# ──── scalar multiplication ───────────────────────────────────────────────────────────────────── #
            
*(a::AbstractNDArray, b::Number) = prod(a, b)
*(a::Number, b::AbstractNDArray) =  *(b, a)