using .Errors: ERR_PERM_IPERM_TYPE, ERR_PERM_IPERM_INV, ERR_PERM_INVALID, ERR_PERM_DIMS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                        PermutedDimsArray                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct PermutedDimsArray{T,N,perm,inv_perm,P<:AbstractArray} <: AbstractNDArray{T,N}
    parent::P

    function PermutedDimsArray{T,N,perm,inv_perm,P}(data::P) where {
        T, N, perm, inv_perm, P <: AbstractArray}
        println(stderr, "Inner constructor called: perm=$perm, inv=$inv_perm")
        # Verify valid input
        same_dim = (isa(perm, NTuple{N,Int}) && isa(inv_perm, NTuple{N,Int}))

        same_dim     || @throw_error ArgumentError ERR_PERM_IPERM_TYPE
        isperm(perm) || @throw_error ArgumentError ERR_PERM_INVALID

        all(d -> inv_perm[perm[d]] == d, 1:N) || @throw_error ArgumentError ERR_PERM_IPERM_INV

        # create view
        new(data)
    end
end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #

const ScalarPermArray{T,perm,inv_perm,P} = PermutedDimsArray{T,0,perm,inv_perm,P}

# ──── constructors ────────────────────────────────────────────────────────────────────────────── #

function PermutedDimsArray(data::AbstractArray{T,N}, perm::NTuple{N,Int}) where {T,N}
    inv_perm = inverse_perm(perm)
    println(stderr, "perm: $perm, invperm: $inv_perm")
    PermutedDimsArray{T,N,perm,inv_perm,typeof(data)}(data)
end

# ══════════════════════════════════ implements AbstractNDArray ══════════════════════════════════ #

# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.parent(a::PermutedDimsArray)    = a.parent
Base.IndexStyle(::PermutedDimsArray) = IndexCartesian()

Base.axes(a::PermutedDimsArray{T,N,perm}) where {T,N,perm} = permute(axes(parent(a)), perm)

# ──── Internal functions ──────────────────────────────────────────────────────────────────────── #

_getElement(a::ScalarPermArray)       = _getElement(parent(a))
_setElement!(a::ScalarPermArray, val) = _setElement!(parent(a), val)

_getElement(a::PermutedDimsArray, i::Int)       = _getElement(a, CartesianIndex(a, i))
_setElement!(a::PermutedDimsArray, val, i::Int) = _setElement!(a, val, CartesianIndex(a, i))

_getElement(a::PermutedDimsArray{T,N,perm,inv_perm},
    I::AllCartesianIndex{N}) where {T,N,perm,inv_perm} = (
    parent(a)[permute(Tuple(I), inv_perm)...]
)

_setElement!(a::PermutedDimsArray{T,N,perm,inv_perm}, val,
    I::AllCartesianIndex{N}) where {T,N,perm,inv_perm} = (
    parent(a)[permute(Tuple(I), inv_perm)...] = val
)

_getElement(a::PermutedDimsArray{T,N,perm,inv_perm},
    indices::NTuple{N,Index}) where {T,N,perm,inv_perm} = (
    parent(a)[permute(indices, inv_perm)...]
)

_setElement!(a::PermutedDimsArray{T,N,perm,inv_perm}, val,
    indices::NTuple{N,Index}) where {T,N,perm,inv_perm} = (
    parent(a)[permute(indices, inv_perm)...] = val
)

# ═════════════════════════════ functions for all PermutedDimsArray ══════════════════════════════ #

# ──── permuted properties ─────────────────────────────────────────────────────────────────────── #

Base.size(a::PermutedDimsArray{T,N,perm}) where {T,N,perm} = permute(size(parent(a)), perm)

function Base.strides(a::PermutedDimsArray{T,N,perm}) where {T,N,perm}
    strds = strides(parent(a))
    ntuple(dim -> strds[perm[dim]], N)
end

fancy_index(a::PermutedDimsArray{T,N,perm,inv_perm}, indices) where {T,N,perm,inv_perm} = (
    fancy_index(parent(a), permute(indices, inv_perm))
)

# ──── permutedims function ────────────────────────────────────────────────────────────────────── #

permutedims(a::AbstractMatrix) = permutedims(a, (2, 1))

# In julia is the permutedims function do not just create PermutedDimsArray but here it's convenient 
permutedims(parent::AbstractArray, perm) = (
    println(stderr, "enter permutedims: ");
    PermutedDimsArray(parent, perm)
)

# ──── utils ───────────────────────────────────────────────────────────────────────────────────── #
            
function check_valid_dims(
    indsP::NTuple{N,AbstractUnitRange}, 
    indsB::NTuple{N,AbstractUnitRange}, 
    perm::NTuple{N,Int}) where N

    for i in eachindex(perm)
        indsP[i] == indsB[perm[i]] || @throw_error DimensionMismatch ERR_PERM_DIMS
    end
    nothing
end

# ──── print function ──────────────────────────────────────────────────────────────────────────── #

function Base.showarg(io::IO, a::PermutedDimsArray{T,N,perm}, toplevel) where {T,N,perm}
    print(io, "Algebra42.PermutedDimsArray(")
    Base.showarg(io, parent(a), false)
    print(io, ", ", perm, ')')
    toplevel && print(io, " with eltype ", eltype(a))
    return nothing
end

# ═══════════════════════════════ compute and execute permutation ════════════════════════════════ #

function inverse_perm(perm::NTuple{N,Int}) where N
    # We search the inverse permutation to obtain the original order of dimension
    # example: perm = (2, 3, 1) the inv_perm = (3, 1, 2) -> inv_perm(perm[d]) = d
    inverse = fill!(Array{Int,1}(undef, N), 0)

    for (i, j) in enumerate(perm)
        ((1 <= j <= N) && inverse[j] == 0) || @throw_error ArgumentError ERR_PERM_INVALID
        inverse[j] = i
    end
    return tuple(inverse...)
end

function permute(I::NTuple{N,Any}, perm::NTuple{N,Int}) where N
    result = ntuple(d -> I[perm[d]], N)
    return result
end

@inline permute(I, perm::AbstractVector{Int}) = permute(I, (perm...,))
