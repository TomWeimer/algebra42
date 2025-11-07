# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       CartesianIndex_42:                                      #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct CartesianIndex_42{N} <: AllCartesianIndex{N}
    I::NTuple{N,Int}
    CartesianIndex_42{N}(index::NTuple{N,Integer}) where {N} = new(index)
end

# ======== Constructor ============================================================== #

# From tupple
CartesianIndex_42(index::NTuple{N,Integer}) where {N} = CartesianIndex_42{N}(index)

# From list of integer
CartesianIndex_42(index::Integer...) = CartesianIndex_42(index)


# ======== Core properties ============================================================== #

# access to indices
Tuple(index::CartesianIndex_42) = index.I

Base.length(::CartesianIndex_42{N}) where {N} = N

Base.eltype(::Type{T}) where {T<:CartesianIndex_42} = eltype(fieldtype(T, :I))

# ======== Conversions ============================================================== #

# A cartesian index of a single number can be used as a number
Base.convert(::Type{T}, index::CartesianIndex_42{1}) where {T<:Number} = convert(T, index[1])

# Try to convert the coordinates to a special type
Base.convert(::Type{T}, index::CartesianIndex_42) where {T<:Tuple} = convert(T, index.I)

# Define conversion from the standard type to the custom type
Base.convert(::Type{CartesianIndex_42{N}}, i::Base.CartesianIndex{N}) where {N} = CartesianIndex_42{N}(i.I)

# Reverse conversion from custom type back to standard type
Base.convert(::Type{Base.CartesianIndex{N}}, i::CartesianIndex_42{N}) where {N} = Base.CartesianIndex{N}(i.I)

Base.checkbounds(a::AbstractArray, I::CartesianIndex_42) = Base.checkbounds(a, convert(Base.CartesianIndex{length(I)}, I))

# ======== Iteration ===================================================================== #

Base.iterate(I::CartesianIndex_42{N}) where N = iterate(I.I)

# ======== Indexing ===================================================================== #

Base.getindex(index::CartesianIndex_42, i::Integer) = index.I[i]

Base.firstindex(index::CartesianIndex_42) = firstindex(index.I)
Base.lastindex(index::CartesianIndex_42)  = lastindex(index.I)

Base.get(A::AbstractArray, I::CartesianIndex_42, default) = get(A, I.I, default)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       in function:                                            #
#_______________________________________________________________________________________________#


# Not same size ( range )
Base.in(x::CartesianIndex_42, r::AbstractRange{<:CartesianIndex_42}) = false

function Base.in(index::CartesianIndex_42{N}, r::AbstractRange{CartesianIndex_42{N}}) where {N}
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

# ======== Print function ===================================================================== #

function Base.show(io::IO, i::CartesianIndex_42)
    print(io, "Algebra42.CartesianIndex_42(")
    join(io, i.I, ", ")
    print(io, ")")
end
