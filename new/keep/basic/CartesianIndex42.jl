include("../other/constant.jl")

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

Base.convert(::Type{CartesianIndex42{N}}, i::AllCartesianIndex{N}) where N = (
    CartesianIndex42{N}(i.I)
)
Base.convert(::Type{AllCartesianIndex{N}}, i::CartesianIndex42{N}) where N = (
    Base.CartesianIndex{N}(i.I)
)

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

Base.firstindex(index::CartesianIndex42) = firstindex(index.I)
Base.lastindex(index::CartesianIndex42)  = lastindex(index.I)

Base.getindex(index::CartesianIndex42, i::Index) = index.I[i]

# TODO: check if we can remove this
Base.get(A::AbstractArray, I::CartesianIndex42, default) = get(A, I.I, default)

Base.checkbounds(a::AbstractArray, I::CartesianIndex42) = Base.checkbounds(a, Tuple(I))

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

