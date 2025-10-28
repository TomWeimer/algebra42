using .Errors: ERR_CARTESIAN_SETINDEX

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       CartesianIndices_42:                                    #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct CartesianIndices_42{N,R <: IndicesRange{N}, Order} <: AbstractArray{ AllCartesianIndex{N}, N}
    indices::R
end

const CartesianIndices42R{N, R} = CartesianIndices_42{N,R, RowOrder}
const CartesianIndices42C{N, R} = CartesianIndices_42{N,R, ColOrder}

# ======== Constructors ======================================================================= #

# Constructor for array
CartesianIndices_42(a::AbstractArray; order=ColOrder) where {N} = CartesianIndices_42(size(a); order=order)

# Constructor for shape tuple
CartesianIndices_42(inds::NTuple{N, Union{<:Integer, OrdinalRange{<:Integer}} }; order=ColOrder) where N = (
    converted_inds = map(_convert, inds);
    CartesianIndices_42{N, typeof(converted_inds), order}(converted_inds) 
)

# ======== Functions to implement AbstractNDArray ============================================= #

# Base functions:

Base.axes(a::CartesianIndices_42)               = map(d -> axes(d, 1), a.indices)
Base.parent(a::CartesianIndices_42)             = a
Base.IndexStyle(a::CartesianIndices_42)         = IndexCartesian()

# Internal functions:

# Scalar
_getElement(a::CartesianIndices_42{0}) = firstindex(a)

# LinearIndices
_getElement(a::CartesianIndices_42,       i::Int) = _getElement(a, _linear_to_cartesian(size(a), i))

# CartesianIndex
_getElement(a::CartesianIndices_42{T, N},  I::AllCartesianIndex{N}) where {T, N} = CartesianIndex_42( map((range, i) -> range[i], a.indices, I) )

# 'Normal' indices
_getElement(a::CartesianIndices_42{T, N},  indices::NTuple{N, Any}) where {T, N} = CartesianIndex_42( map((range, i) -> range[i], a.indices, indices) )

# setindex errors
_setElement!(::CartesianIndices_42{0}, val)                                = @throw_error MethodError ERR_CARTESIAN_SETINDEX
_setElement!(::CartesianIndices_42, val, i::Int)                           = @throw_error MethodError ERR_CARTESIAN_SETINDEX
_setElement!(::CartesianIndices_42, val, I::AllCartesianIndex)             = @throw_error MethodError ERR_CARTESIAN_SETINDEX
_setElement!(::CartesianIndices_42, val, indices::NTuple{N, Any}) where N  = @throw_error MethodError ERR_CARTESIAN_SETINDEX

# ======== Functions for all CartesianIndices_42 ================================================== #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                            indexing:                                          #
#_______________________________________________________________________________________________#

Base.eachindex(::IndexCartesian, a::CartesianIndices_42) = a

Base.first(iter::CartesianIndices_42) = CartesianIndex_42(map(first, iter.indices))
Base.step(iter::CartesianIndices_42)  = CartesianIndex_42(map(step, iter.indices))
Base.last(iter::CartesianIndices_42)  = CartesianIndex_42(map(last, iter.indices))


Base.in(i::CartesianIndex_42{N}, r::CartesianIndices_42)    where N = false # Wrong dimensions
Base.in(i::CartesianIndex_42{N}, r::CartesianIndices_42{N}) where N = all(map(in, i.I, r.indices))

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       iteration:                                              #
#_______________________________________________________________________________________________#

function Base.iterate(iter::CartesianIndices_42)
    iterfirst = first(iter)
    if !all(map(in, iterfirst.I, iter.indices)) # call in(Int, OrdinalRangeInt)
        return nothing
    end
    iterfirst, iterfirst
end

function Base.iterate(iter::CartesianIndices42C, state::Base.AbstractCartesianIndex)
    valid, I = _inc(state.I, iter.indices)
    valid || return nothing
    return CartesianIndex_42(I...), CartesianIndex_42(I...)
end

function Base.iterate(iter::CartesianIndices42R, state::Base.AbstractCartesianIndex)
    valid, I = _inc_row_order(state.I, iter.indices)
    valid || return nothing
    return CartesianIndex_42(I...), CartesianIndex_42(I...)
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                increment functions:                                           #
#_______________________________________________________________________________________________#

# from shape tuple
inc(state::AllCartesianIndex{N}, shape::NTuple{N,Int}; order=ColOrder) where {N} = (
    inc(state, map(_convert, shape, order = order))
)

# from range tuple
function inc(state::AllCartesianIndex{N}, indices::NTuple{N,OrdinalRangeInt}; order=ColOrder) where N
    valid, I = (order == ColOrder) ? _inc(state.I, indices) : _inc_row_order(state.I, indices)
    valid || return nothing
    return typeof(state)(I)
end

# ======== Increment Column Order ============================================================= #

function _inc(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _inc(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[1] != last(ranges[1]), (indices[1] + step(ranges[1]),)
end

function _inc(indices::Tuple{Int,Int,Vararg{Int}}, ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
    idx, range = indices[1], ranges[1]

    if idx != last(range)
        # Incremenet the actual index
        return true, (idx + step(range), Base.tail(indices)...)
    end
    # Reset the index and increment the next index
    valid, nextIndices = _inc(Base.tail(indices), Base.tail(ranges))

    return valid, (first(range), nextIndices...)
end

# ======== Increment Row Order ================================================================ #

function _inc_row_order(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _inc_row_order(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[end] != last(ranges[end]), (indices[end] + step(ranges[end]),)
end

function _inc_row_order(indices::Tuple{Int,Int,Vararg{Int}}, ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
    idx, range = indices[end], ranges[end]

    if idx != last(range)
        # Incremenet the actual index
        return true, (Base.front(indices)..., idx + step(range))
    end
    # Reset the index and increment the next index
    valid, nextIndices = _inc_row_order(Base.front(indices), Base.front(ranges))

    return valid, (nextIndices..., first(range))
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                decrement functions:                                           #
#_______________________________________________________________________________________________#

# from shape tuple
dec(state::AllCartesianIndex{N}, shape::NTuple{N,Int}; order=ColOrder) where {N} = (
    dec(state, map(_convert, shape, order = order))
)

# from range tuple
function dec(state::AllCartesianIndex{N}, indices::NTuple{N,OrdinalRangeInt}; order=ColOrder) where N
    valid, I = (order == ColOrder) ? _dec(state.I, indices) : _dec_row_order(state.I, indices)
    valid || return nothing
    return typeof(state)(I)
end

# ======== Decrement Column Order ============================================================= #

function _dec(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _dec(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[1] != first(ranges[1]), (indices[1] - step(ranges[1]),)
end

function _dec(indices::Tuple{Int,Int,Vararg{Int}}, ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
    idx, range = indices[1], ranges[1]

    if idx != first(range)
        # Incremenet the actual index
        return true, (idx - step(range), Base.tail(indices)...)
    end
    # Reset the index and increment the next index
    valid, nextIndices = _dec(Base.tail(indices), Base.tail(ranges))

    return valid, (last(range), nextIndices...)
end

# ======== Decrement Row Order ================================================================ #

function _dec_row_order(::Tuple{}, ::Tuple{})
    return  false, ()
end

function _dec_row_order(indices::Tuple{Int}, ranges::Tuple{OrdinalRangeInt})
    return indices[end] != first(ranges[end]), (indices[end] - step(ranges[end]),)
end

function _dec_row_order(indices::Tuple{Int,Int,Vararg{Int}}, ranges::Tuple{OrdinalRangeInt, OrdinalRangeInt, Vararg{OrdinalRangeInt}})
    idx, range = indices[end], ranges[end]

    if idx != first(range)
        # Incremenet the actual index
        return true, (Base.front(indices)..., idx - step(range))
    end
    # Reset the index and increment the next index
    valid, nextIndices = _dec_row_order(Base.front(indices), Base.front(ranges))

    return valid, (nextIndices..., last(range))
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                      print functions:                                         #
#_______________________________________________________________________________________________#

Base.show(io::IO, ::MIME"text/plain", iter::CartesianIndices_42) = show(io, iter)

function Base.show(io::IO, iter::CartesianIndices_42)
    print(io, "Algebra42.CartesianIndices_42(")
    show(io, map(_form_index, iter.indices))
    print(io, ")")
end

_form_index(i) = i
_form_index(i::Base.OneTo) = i.stop
    
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                        converting:                                            #
#_______________________________________________________________________________________________#

_convert(sz::Integer)           = Base.oneto(sz)
_convert(sz::AbstractUnitRange) = first(sz):last(sz)
_convert(sz::OrdinalRange)      = first(sz):step(sz):last(sz)