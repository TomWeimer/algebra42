include("CartesianIndex42.jl")

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

# TODO: check if we can remove some constructors

CartesianIndices42(::Tuple{}; order=ColOrder) = CartesianIndices42{0,typeof(()), order}(())

CartesianIndices42(a::AbstractArray; order=ColOrder) = CartesianIndices42(axes(a); order=order)

CartesianIndices42(I::AllCartesianIndex) = CartesianIndices42(Tuple(I))

CartesianIndices42(inds::NTuple{N,OrdinalRange{<:Integer, <:Integer}}) where N = (
    indices = map(r->convert(OrdinalRangeInt, r), inds);
    CartesianIndices42{N, typeof(indices)}(indices)
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

# ═══════════════════════════════════════════ indexing ═══════════════════════════════════════════ #

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

