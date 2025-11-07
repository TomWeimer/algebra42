# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                        CartesianIndex42:                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct CartesianIndex42{N} <: AllCartesianIndex{N}
    I::NTuple{N,Int}
    CartesianIndex42{N}(index::NTuple{N,Int}) where N = new(index)
end

# ──── Constructors: ───────────────────────────────────────────────────────────────────────────── #
            
CartesianIndex42(index::Index...) = CartesianIndex42(index)
CartesianIndex42(index::IndicesInt{N}) where N = CartesianIndex42{N}(index)

CartesianIndex42(a::AbstractArray, i::Index; order = ColOrder) = (
    CartesianIndex42(from_linear_to_cartesian(size(a), i; order = order))
)

# ──── core properties: ────────────────────────────────────────────────────────────────────────── #

Core.Tuple(index::CartesianIndex42) = index.I

Base.length(::CartesianIndex42{N}) where N = N
Base.eltype(::Type{T}) where {T<:CartesianIndex42} = eltype(fieldtype(T, :I))

# ──── conversions: ────────────────────────────────────────────────────────────────────────────── #
            
Base.convert(::Type{T}, I::CartesianIndex42{1}) where T <: Number = convert(T, I[1])
Base.convert(::Type{T}, I::CartesianIndex42)    where T <: Tuple  = convert(T, Tuple(I))

Base.convert(::Type{CartesianIndex42}, i::CartesianIndex) = CartesianIndex42(i.I)

Base.convert(::Type{CartesianIndex}, i::CartesianIndex42) = CartesianIndex(i.I)

function from_linear_to_cartesian(sizeA::Dims{N}, i::Index; order=ColOrder) where N
    # if empty or scalar then return
    N == 0 && return ()
    inds = Array{Int, 1}(undef, N)
    L_R = i - 1

    iteration_range = order == ColOrder ? (1:N) : (N:-1:1)

    for k in iteration_range
        i_k = L_R % sizeA[k]
        L_R = div(L_R, sizeA[k])
        inds[k] = i_k + 1
    end
    return tuple(inds...)
end

# ═════════════════════════════════════ Iteration & indexing ═════════════════════════════════════ #
            
Base.iterate(I::CartesianIndex42) = iterate(I.I)
Base.iterate(I::CartesianIndex42, state::Index) = iterate(I.I, state)

Base.firstindex(index::CartesianIndex42) = firstindex(index.I)
Base.lastindex(index::CartesianIndex42)  = lastindex(index.I)

Base.getindex(index::CartesianIndex42, i::Index) = index.I[i]

# TODO: check if we can remove this
Base.get(A::AbstractArray, I::CartesianIndex42, default) = get(A, I.I, default)

# ──── bounds check ────────────────────────────────────────────────────────────────────────────── #

@inline function Base.checkbounds(::Type{Bool}, a::AbstractArray, I::CartesianIndex42)
     Base.checkbounds_indices(Bool, axes(a), (I,))
end

@inline function Base.checkbounds_indices(::Type{Bool}, a::AbstractArray, I::CartesianIndex42)
     Base.checkbounds_indices(Bool, axes(a), (I,))
end

@inline function Base.checkbounds_indices(
    ::Type{Bool}, inds::Tuple, I::Tuple{CartesianIndex42,Vararg})

    isempty(I) && return true

    dims = length(I[1])
    cartIndex = convert(CartesianIndex, I[1])

    return (
        checkindex(Bool, inds[1:dims], cartIndex) & 
        Base.checkbounds_indices(Bool, inds[dims+1:end], Base.tail(I))
    )
end

# Because Base do not use AbstractCartesianIndex we need to redefine some function....
# Those are exact copy of base functions but with out CartesianIndex42 type

@inline  Base.to_indices(A, I::Tuple{Vararg{Union{Integer, CartesianIndex42}}}) = to_indices(A, (), I)

@inline function  Base.to_indices(A, inds, I::Tuple{CartesianIndex42{N}, Vararg}) where N
    _, indstail = Base.IteratorsMD.split(inds, Val(N))
    (map(Base.Fix1(Base.to_index, A), I[1].I)..., to_indices(A, indstail, Base.tail(I))...)
end

@inline function Base.to_indices(A, inds, I::Tuple{AbstractArray{CartesianIndex42{N}}, Vararg}) where N
    _, indstail = Base.IteratorsMD.split(inds, Val(N))
    (Base.to_index(A, I[1]), to_indices(A, indstail, Base.tail(I))...)
end


# to_indices(A, I::Tuple) = (@inline; to_indices(A, axes(A), I))

#  # When used as indices themselves, CartesianIndices can simply become its tuple of ranges
# @inline function to_indices(A, inds, I::Tuple{CartesianIndices{N}, Vararg}) where N
#     _, indstail = split(inds, Val(N))
#     (map(Fix1(to_index, A), I[1].indices)..., to_indices(A, indstail, tail(I))...)
# end
# Base.to_indices(A, I::Tuple{CartesianIndex42, Vararg}) = (@inline; to_indices(A, axes(A), I))

# ──── in function ─────────────────────────────────────────────────────────────────────────────── #

Base.in(x::CartesianIndex42, r::AbstractRange{CartesianIndex42}) = false # wrong nb of elements

function Base.in(index::CartesianIndex42{N}, r::AbstractRange{CartesianIndex42{N}}) where N
    isempty(r) && return false

    first_idx = first(r)
    step_idx  = step(r)
    last_idx  = last(r)
    
    # For each dimension, find the first non-zero step dimension.
    # That dimension determines the position `n` in the range.
    for i in 1:N
        
        si = step_idx[i]
        if !iszero(si)
            # Try to find the index position along that dimension
            rng = first_idx[i]:si:last_idx[i]
            n = findfirst(==(index[i]), rng)
            return !isnothing(n) && r[n] == index
        end
    end
   
    # If all step dims are zero, range is a repeated value → check equality
    return index == first_idx
end

# ──── print function ──────────────────────────────────────────────────────────────────────────── #

function Base.show(io::IO, i::CartesianIndex42)
    print(io, "Algebra42.CartesianIndex42(")
    join(io, i.I, ", ")
    print(io, ")")
end

using .Errors: ERR_NOT_STORED_VALUE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                       CartesianIndices42:                                        #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct CartesianIndices42{N, R <:IndicesRange{N}, Order} <: AbstractNDArray{AllCartesianIndex{N}, N}
    indices::R
end

# ──── Constructors ────────────────────────────────────────────────────────────────────────────── #

CartesianIndices42(::Tuple{}; order=ColOrder) = CartesianIndices42{0,typeof(()), order}(())

CartesianIndices42(a::AbstractArray; order=ColOrder) = CartesianIndices42(axes(a); order=order)

CartesianIndices42(I::AllCartesianIndex; order=ColOrder) = CartesianIndices42(Tuple(I); order=order)

CartesianIndices42(inds::NTuple{N,OrdinalRange{<:Integer, <:Integer}}; order = ColOrder) where N = (
    indices = map(r->convert(OrdinalRangeInt, r), inds);
    CartesianIndices42{N, typeof(indices), order}(indices)
)

CartesianIndices42(inds::NTuple{N,Union{<:Integer,OrdinalRange{<:Integer}}};
                   order = ColOrder) where N = (
    converted_inds = map(_convert, inds);
    CartesianIndices42{N, typeof(converted_inds), order}(converted_inds) 
)

# ═════════════════════════════════ implements AbstractNDArray: ══════════════════════════════════ #

# ──── Base functions: ─────────────────────────────────────────────────────────────────────────── #
            
Base.axes(a::CartesianIndices42)               = map(d -> axes(d, 1), a.indices)
Base.parent(a::CartesianIndices42)             = a
Base.IndexStyle(::CartesianIndices42)         = IndexCartesian()

# ──── Internal functions: ─────────────────────────────────────────────────────────────────────── #

_getElement(a::CartesianIndices42{0}) = firstindex(a)

_getElement(a::CartesianIndices42{N, R, Order}, i::Index) where {N, R, Order} = (
    _getElement(a, CartesianIndex42(a, i; order=Order))
)

_getElement(a::CartesianIndices42{N}, I::Indices{N}) where N = (
    CartesianIndex42( map((range, i) -> range[i], a.indices, I) )
)

# setindex errors:
# ---------------
_setElement!(::CartesianIndices42{0}, val)        = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::CartesianIndices42, val, i::Index) = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::CartesianIndices42, v, I::Indices) = @throw_error ArgumentError ERR_NOT_STORED_VALUE

# ═════════════════════════════════════════ conversions ══════════════════════════════════════════ #

_convert(sz::Integer)           = Base.oneto(sz)
_convert(sz::AbstractUnitRange) = first(sz):last(sz)
_convert(sz::OrdinalRange)      = first(sz):step(sz):last(sz)

# ═══════════════════════════════════ indexing and iteration ═════════════════════════════════════ #

# ──── indexing ────────────────────────────────────────────────────────────────────────────────── #

# TODO: see if we can remove it
Base.getindex(a::CartesianIndices42{N,R}, 
              I::Vararg{Union{OrdinalRange{<:Integer,<:Integer},Colon},N}) where {N,R} = (
    @boundscheck checkbounds(a, I...);
    indices = map((range, i) ->  @inbounds range[i], a.indices, I);
    CartesianIndices42(indices)
)

Base.eachindex(::IndexCartesian, a::CartesianIndices42) = a
Base.eachindex(::IndexCartesian, A::AbstractNDArray)     = CartesianIndices42(A)

Base.first(iter::CartesianIndices42) = CartesianIndex42(map(first, iter.indices))
Base.step(iter::CartesianIndices42)  = CartesianIndex42(map(step, iter.indices))
Base.last(iter::CartesianIndices42)  = CartesianIndex42(map(last, iter.indices))


Base.in(i::CartesianIndex42{N}, r::CartesianIndices42)    where N = false # Wrong dimensions
Base.in(i::CartesianIndex42{N}, r::CartesianIndices42{N}) where N = all(map(in, i.I, r.indices))

# ──── iteration ───────────────────────────────────────────────────────────────────────────────── #

function Base.iterate(iter::CartesianIndices42)
    iterfirst = first(iter)
    if !all(map(in, iterfirst.I, iter.indices)) # call in(Int, OrdinalRangeInt)
        return nothing
    end
    iterfirst, iterfirst
end

function Base.iterate(iter::CartesianIndices42{N, R, Order}, state::AllCartesianIndex{N}) where {
    N, R, Order
} 
    I = inc(state, iter.indices, order=ColOrder)
    isnothing(I) && return nothing
    return I, I
end


# ──── increments functions ────────────────────────────────────────────────────────────────────── #            

inc(state::AllCartesianIndex{N}, shape::Dims{N}; order=ColOrder) where N = (
    inc(state, map(_convert, shape), order=order)
)

function inc(state::AllCartesianIndex{N}, indices::IndicesRange{N}; order=ColOrder) where N
    valid, I = (order == ColOrder) ? _inc(state.I, indices) : _inc_row_order(state.I, indices)
    valid || return nothing
    return typeof(state)(I)
end

# ──── inc column major ────────────────────────────────────────────────────────────────────────── #

function _inc(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _inc(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[1] != last(ranges[1]), (indices[1] + step(ranges[1]),)
end

function _inc( indices::Tuple{Int,Int,Vararg{Int}}, 
               ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})

    idx, range = indices[1], ranges[1]

    if idx != last(range)
        # Incremenet the actual index
        return true, (idx + step(range), Base.tail(indices)...)
    end
    # Reset the index and increment the next index
    valid, nextIndices = _inc(Base.tail(indices), Base.tail(ranges))

    return valid, (first(range), nextIndices...)
end

# ──── inc row major ───────────────────────────────────────────────────────────────────────────── #
            
function _inc_row_order(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _inc_row_order(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[end] != last(ranges[end]), (indices[end] + step(ranges[end]),)
end

function _inc_row_order( indices::Tuple{Int,Int,Vararg{Int}}, 
                         ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
                         
    idx, range = indices[end], ranges[end]

    if idx != last(range)
        # Incremenet the actual index
        return true, (Base.front(indices)..., idx + step(range))
    end
    # Reset the index and increment the next index
    valid, nextIndices = _inc_row_order(Base.front(indices), Base.front(ranges))

    return valid, (nextIndices..., first(range))
end

# ──── decrements functions ────────────────────────────────────────────────────────────────────── #
            
dec(state::AllCartesianIndex{N}, shape::Dims{N}; order=ColOrder) where N = (
    dec(state, map(_convert, shape), order = order)
)

function dec(state::AllCartesianIndex{N}, indices::IndicesRange{N}; order=ColOrder) where N
    valid, I = (order == ColOrder) ? _dec(state.I, indices) : _dec_row_order(state.I, indices)
    valid || return nothing
    return typeof(state)(I)
end

# ──── dec column major ────────────────────────────────────────────────────────────────────────── #
            
function _dec(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _dec(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[1] != first(ranges[1]), (indices[1] - step(ranges[1]),)
end

function _dec( indices::Tuple{Int,Int,Vararg{Int}}, 
               ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}} )

    idx, range = indices[1], ranges[1]

    if idx != first(range)
        # Incremenet the actual index
        return true, (idx - step(range), Base.tail(indices)...)
    end
    # Reset the index and increment the next index
    valid, nextIndices = _dec(Base.tail(indices), Base.tail(ranges))

    return valid, (last(range), nextIndices...)
end

# ──── dec row major ───────────────────────────────────────────────────────────────────────────── #
            
function _dec_row_order(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _dec_row_order(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[end] != first(ranges[end]), (indices[end] - step(ranges[end]),)
end

function _dec_row_order( indices::Tuple{Int,Int,Vararg{Int}}, 
                         ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
                         
    idx, range = indices[end], ranges[end]

    if idx != first(range)
        # Incremenet the actual index
        return true, (Base.front(indices)..., idx - step(range))
    end
    # Reset the index and increment the next index
    valid, nextIndices = _dec_row_order(Base.front(indices), Base.front(ranges))

    return valid, (nextIndices..., last(range))
end

# ──── print function ──────────────────────────────────────────────────────────────────────────── #

Base.show(io::IO, ::MIME"text/plain", iter::CartesianIndices42) = show(io, iter)

function Base.show(io::IO, iter::CartesianIndices42)
    print(io, "Algebra42.CartesianIndices42(")
    show(io, map(_form_index, iter.indices))
    print(io, ")")
end

_form_index(i) = i
_form_index(i::Base.OneTo) = i.stop
