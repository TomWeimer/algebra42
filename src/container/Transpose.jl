struct Transpose{T} <: AbstractNDArray{T, 2}
    parent::AbstractNDArray{T, 2}

    Transpose(A::AbstractNDArray{T, 2}) where{T} = new{T}(A)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Functions to Overload:                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# To create a class that behave like a AbstractNDArray, you only need to overload the following functions:

# Return the indexStyle used by the array
Base.IndexStyle(a::Transpose) = IndexStyle(a.parent)

# Return the shape of the array
Base.size(a::Transpose) = reverse(size(a.parent))

# Return the shape of the array
Base.axes(a::Transpose) = reverse(axes(a.parent))

# Return the parent array, or self if no parent exists
Base.parent(a::Transpose) = a.parent

Base.strides(a::Transpose) = reverse(strides(a.parent))

Base.isassigned(a::Transpose, i::Int, j::Int) = Base.isassigned(a.parent, j, i)

# Get and set elements:
# ---------------------

# LinearIndices
function _getElement(a::Transpose{T}, i::Int) where T
    # linear index i in transposed shape
    rowsT, colsT = size(a)  # size after transpose
    rowT = (i - 1) % rowsT + 1
    colT = (i - 1) ÷ rowsT + 1

    # map to parent
    rowP = colT
    colP = rowT
    return a.parent[rowP, colP]
end

function _setElement!(a::Transpose{T}, val::T, i::Int) where T
    rowsT, colsT = size(a)
    rowT = (i - 1) % rowsT + 1
    colT = (i - 1) ÷ rowsT + 1
    rowP = colT
    colP = rowT
    a.parent[rowP, colP] = val
end

# CartesianIndex
_getElement(a::Transpose{DType}, I::Base.AbstractCartesianIndex{2}) where {DType} = _getElement(a, Tuple(I))
_setElement!(a::Transpose{DType}, val::DType, I::Base.AbstractCartesianIndex{2}) where {DType} = _setElement!(a, val, Tuple(I))

# 'Normal' indices
_getElement(a::Transpose{T}, idx::NTuple{2, Any})          where T = a.parent[idx[2], idx[1]]
_setElement!(a::Transpose{T}, val::T, idx::NTuple{2, Any}) where T = a.parent[idx[2], idx[1]] = val