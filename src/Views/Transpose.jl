# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                            Transpose                                             #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct Transpose{T} <: AbstractNDArray{T, 2}
    parent::AbstractNDArray{T, 2}

    Transpose(A::AbstractNDArray{T, 2}) where{T} = new{T}(A)
end

# ══════════════════════════════════ implements AbstractNDArray ══════════════════════════════════ #
            
# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.parent(a::Transpose)     = a.parent
Base.IndexStyle(a::Transpose) = (IndexStyle ∘ parent)(a)
Base.axes(a::Transpose)       = (reverse ∘ axes ∘ parent)(a)

# ──── Internal functions ──────────────────────────────────────────────────────────────────────── #
            
# LinearIndices
_getElement(a::Transpose{T}, i::Int) where T = (
    (rowP, colP) = pos_from_linear_index(a, i);
    a.parent[rowP, colP]
)

_setElement!(a::Transpose, val, i::Int) = (
    (rowP, colP) = pos_from_linear_index(a, i);
    a.parent[rowP, colP] = val
)

# CartesianIndex
_getElement(a::Transpose{DType}, I::AllCartesianIndex{2}) where {DType} = (
    _getElement(a, Tuple(I))
)

_setElement!(a::Transpose{DType}, val::DType, I::AllCartesianIndex{2}) where {DType} = (
    _setElement!(a, val, Tuple(I))
)

# 'Normal' indices
_getElement(a::Transpose{T}, idx::NTuple{2, Index}) where T = (
    a.parent[idx[2], idx[1]]
)

_setElement!(a::Transpose, val, idx::NTuple{2, Index}) = (
    a.parent[idx[2], idx[1]] = val
)

# ═════════════════════════════════ functions for all Transpose ══════════════════════════════════ #

Base.size(a::Transpose)    = (reverse ∘ size ∘ parent)(a)
Base.strides(a::Transpose) = (reverse ∘ strides ∘ parent)(a)

Base.isassigned(a::Transpose, i::Int, j::Int) = Base.isassigned(parent(a), j, i)

# Return column and rows of the parent from linear index 
function pos_from_linear_index(a::Transpose, i::Int)
    rowsT, colsT = size(a)

    rowT = (i - 1) % rowsT + 1
    colT = (i - 1) ÷ rowsT + 1

    # map to parent
    rowP = colT
    colP = rowT

    return (rowP, colP)
end