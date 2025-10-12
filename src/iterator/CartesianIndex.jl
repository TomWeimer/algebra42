# MultiDimensional index
struct CartesianIndex_42{N} <: Base.AbstractCartesianIndex{N}
    I::NTuple{N,Int}
    CartesianIndex_42{N}(index::NTuple{N,Integer}) where {N} = new(index)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructors:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# From tupple
CartesianIndex_42(index::NTuple{N,Integer}) where {N} = CartesianIndex_42{N}(index)

# From list of integer
CartesianIndex_42(index::Integer...) = CartesianIndex_42(index)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Utils:                                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# access to index tuple
Tuple(index::CartesianIndex_42) = index.I


Base.length(::CartesianIndex_42{N}) where {N} = N

# eltype(CartesianIndex_42{N}) ⟹ eltype(NTuple{N,Int}) ⟹ Int
Base.eltype(::Type{T}) where {T<:CartesianIndex_42} = eltype(fieldtype(T, :I))

# print
function show(io::IO, i::CartesianIndex_42)
    print(io, "Algebra42.CartesianIndex_42(")
    join(io, i.I, ", ")
    print(io, ")")
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Indexing:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# []
Base.getindex(index::CartesianIndex_42, i::Integer) = index.I[i]

# first
Base.firstindex(index::CartesianIndex_42) = firstindex(index.I)

# last
Base.lastindex(index::CartesianIndex_42) = lastindex(index.I)

# get
Base.get(A::AbstractArray, I::CartesianIndex_42, default) = get(A, I.I, default)

# Not same size ( range )
Base.in(x::CartesianIndex_42, r::AbstractRange{<:CartesianIndex_42}) = false

function Base.in(index::CartesianIndex_42{N}, r::AbstractRange{CartesianIndex_42{N}}) where {N}
    isempty(r) && return false
    first, step, last = first(r), step(r), last(r)
    # The n-th element of the range is a CartesianIndex_42
    # whose elements are the n-th along each dimension
    # Find the first dimension along which the index is changing,
    # so that n may be uniquely determined
    for i in 1:N
        # step[i] is zero then continue
        iszero(step[i]) && continue
        # obtain the 
        n = findfirst(==(index[i]), first[i]:step[i]:last[i])
        isnothing(n) && return false
        return r[n] == index
    end
    # if the step is zero, the elements are identical, so compare with the first
    return index == first
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Conversions:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# A cartesian index of a single number can be used as a number
convert(::Type{T}, index::CartesianIndex_42{1}) where {T<:Number} = convert(T, index[1])

# Try to convert the coordinates to a special type
convert(::Type{T}, index::CartesianIndex_42) where {T<:Tuple} = convert(T, index.I)

function Base.getindex(A::AbstractArray, I::Base.AbstractCartesianIndex)
    # The splat operator (unpacking/deconstructing) '...' is the key.
    # It converts the AbstractCartesianIndex into a sequence of integer arguments,
    # which is the standard way to index an array.
    return Base.getindex(A, I.I...)
end

function Base.setindex!(A::AbstractArray, v, I::Base.AbstractCartesianIndex)
    # The new value 'v' is the second argument.
    # The splat operator '...' again unpacks the index object I into separate integer arguments.
    Base.setindex!(A, v, I.I...)
end