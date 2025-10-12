const OrdinalRangeInt = OrdinalRange{Int,Int}

# Span the cartesian space
struct CartesianIndices_42{N,R<:NTuple{N,OrdinalRangeInt}} <: AbstractArray{CartesianIndex_42{N},N}
    indices::R
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructors:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Empty constructor
CartesianIndices_42_42(::Tuple{}) = CartesianIndices_42_42{0,typeof(())}(())

# Default constructor
CartesianIndices_42(inds::NTuple{N,Union{<:Integer,OrdinalRange{<:Integer}}}) where {N} =
    CartesianIndices_42(map(_convert, inds))

# Convert all entries of the tuple to the same basic Int type
function CartesianIndices_42(inds::NTuple{N,OrdinalRange{<:Integer,<:Integer}}) where {N}
    indices = map(r -> convert(OrdinalRangeInt, r), inds)
    CartesianIndices_42{N,typeof(indices)}(indices)
end

# convert integer to range from 1 to sz 
_convert(sz::Integer) = Base.oneto(sz)

# convert ranges to Int type 
_convert(sz::AbstractUnitRange) = first(sz):last(sz)
_convert(sz::OrdinalRange) = first(sz):step(sz):last(sz)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Utils:                                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Julia's map function generally attempts to preserve the "outer" container type of the input collection
Base.size(iter::CartesianIndices_42)  = map(length, iter.indices) # ->  R <: NTuple{N,OrdinalRangeInt}

Base.last(iter::CartesianIndices_42)  = CartesianIndex_42(map(last, iter.indices))

Base.first(iter::CartesianIndices_42) = CartesianIndex_42(map(first, iter.indices))

Base.step(iter::CartesianIndices_42)  = CartesianIndex_42(map(step, iter.indices))

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Getters:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function Base.getindex(iter::CartesianIndices_42{N,R}, I::Vararg{Int,N}) where {N,R}
    # Check bounds at a higher level
    @boundscheck checkbounds(iter, I...)
    index = map(iter.indices, I) do r, i
        # Assume corectness for all index
        @inbounds getindex(r, i)
    end
    CartesianIndex_42(index)
end

function Base.getindex(iter::CartesianIndices_42{N,R}, I::Vararg{Union{OrdinalRange{<:Integer,<:Integer},Colon},N}) where {N,R}
     # Check bounds at a higher level
    @boundscheck checkbounds(iter, I...)
    indices = map(iter.indices, I) do r, i
        # Assume corectness for all index
        @inbounds getindex(r, i)
    end
    CartesianIndices_42(indices)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Iterate:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function Base.iterate(iter::CartesianIndices_42)
    iterfirst = first(iter)
    if !all(map(in, iterfirst.I, iter.indices)) # call in(Int, OrdinalRangeInt)
        return nothing
    end
    iterfirst, iterfirst
end

function Base.iterate(iter::CartesianIndices_42, state::CartesianIndex_42)
    valid, I = __inc(state.I, iter.indices)
    valid || return nothing
    return CartesianIndex_42(I...), CartesianIndex_42(I...)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Increment/Decrement:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# increment & carry
function inc(state, indices)
    _, I = __inc(state, indices)
    return CartesianIndex_42(I...)
end

# This code implements the recursive, column-major iteration for CartesianIndices_42. 
# By explicitly using a valid::Bool flag instead of returning Union{Nothing, Tuple}, 
# it eliminates type instability in the core __inc loop, which boosts runtime performance.

#  Base Case (Empty Tuple) -> return invalid/overflow
__inc(::Tuple{}, ::Tuple{}) = false, ()

# Single-Dimension Case (The Stop/Start)
function __inc(state::Tuple{Int}, indices::Tuple{OrdinalRangeInt})
    range = indices[1]
    I = state[1] + step(range)
    valid = state[1] != last(range)
    return valid, (I,)
end

# Recursive Step (The "Carry" Logic)
function __inc(state::Tuple{Int,Int,Vararg{Int}}, indices::Tuple{OrdinalRangeInt,OrdinalRangeInt,Vararg{OrdinalRangeInt}})
    range = indices[1]
    # if we are not at the end of a dimension just increase it, and return the index is found
    if state[1] != last(range)
        I = state[1] + step(range)
        return true, (I, Base.tail(state)...)
    end
    # Otherwise, increment the dimension and then continue with the next
    valid, Itail = __inc(Base.tail(state), Base.tail(indices))
    return valid, (first(range), Itail...)
end


# decrement & carry
function dec(state, indices)
    _, I = __dec(state, indices)
    return CartesianIndex_42(I...)
end

#  Base Case (Empty Tuple) -> return invalid/overflow
@inline __dec(::Tuple{}, ::Tuple{}) = false, ()

# Inverse of __inc
function __dec(state::Tuple{Int}, indices::Tuple{OrdinalRangeInt})
    range = indices[1]
    I = state[1] - step(range)
    valid = state[1] != first(range)
    return valid, (I,)
end

# Inverse of __inc
function __dec(state::Tuple{Int,Int,Vararg{Int}}, indices::Tuple{OrdinalRangeInt,OrdinalRangeInt,Vararg{OrdinalRangeInt}})
    range = indices[1]
    I = state[1] - step(range)
    if state[1] != first(range)
        return true, (I, Base.tail(state)...)
    end
    valid, I = __dec(Base.tail(state), Base.tail(indices))
    return valid, (last(range), I...)
end


 function show(io::IO, iter::CartesianIndices_42)
        print(io, "Algebra42.CartesianIndices_42(")
        show(io, map(_form_index, iter.indices))
        print(io, ")")
    end
    _form_index(i) = i
    _form_index(i::Base.OneTo) = i.stop
    show(io::IO, ::MIME"text/plain", iter::CartesianIndices_42) = show(io, iter)


    # Not same size ( Cartesian Indices )
Base.in(i::CartesianIndex_42, r::CartesianIndices_42) = false

# With same size ( Cartesian Indices )
Base.in(i::CartesianIndex_42{N}, r::CartesianIndices_42{N}) where {N} = all(map(in, i.I, r.indices))