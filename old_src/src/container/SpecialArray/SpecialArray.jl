using .Errors: ERR_NOT_STORED_VALUE, ERR_SCALAR_DIAG, ERR_IS_NOT_SQUARE, ERR_DIAGONAL_DIM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       AbstractDiagonalArray:                                  #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type AbstractDiagonalArray{T, N} <: AbstractNDArray{T, N} end

const AbstractDiagonalMatrix{T} = AbstractDiagonalArray{T, 2}

# ======== Functions to implement AbstractNDArray ============================================================== #

Base.axes(a::AbstractDiagonalArray{T, N}) where {T, N}  = (d = diagonal_length(a); ntuple(_ -> 1:d, N))
Base.parent(a::AbstractDiagonalArray)             = a
Base.IndexStyle(::AbstractDiagonalArray)          = IndexCartesian()

# Internal functions:
_getElement(a::AbstractDiagonalArray{T, N},  i::Int)                    where {T, N} = __getElement(a, i)
_getElement(a::AbstractDiagonalArray{T, N},  I::AllCartesianIndex{N})   where {T, N} = __getElement(a, I)
_getElement(a::AbstractDiagonalArray{T, N},  indices::NTuple{N, Index}) where {T, N} = __getElement(a, indices)

_getElement(::AbstractDiagonalArray{T, 0})       where T  = @throw_error ArgumentError ERR_SCALAR_DIAG
_setElement!(::AbstractDiagonalArray{T, 0}, val) where T  = @throw_error ArgumentError ERR_SCALAR_DIAG

_setElement!(::AbstractDiagonalArray, val, i::Int)                                     = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::AbstractDiagonalArray{T, N}, val, I::AllCartesianIndex{N}) where {T, N} = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::AbstractDiagonalArray{T, N}, val, indices::NTuple{N, Any}) where {T, N} = @throw_error ArgumentError ERR_NOT_STORED_VALUE

# ======== Functions for all AbstractDiagonalArray ============================================================== #

 # If the array is not a matrix we convert to cartesian index
__getElement(a::AbstractDiagonalArray{T, N}, i::Int) where {T, N} = __getElement(a, from_linear_to_cartesian(size(a), i))

 # And if the array is a matrix we keep the linear index
__getElement(a::AbstractDiagonalMatrix{T}, i::Int) where T = (
    diagLength = diagonal_length(a);
    isOnDiag(a, i, diagLength) ? getDiagValue(a, i, diagLength) : zero(T)
)

__getElement(a::AbstractDiagonalArray{T, N}, idx::Union{AllCartesianIndex{N}, NTuple{N, Index}}) where {T, N} = (
    dim1 = first(idx);
    isOnDiag(a, idx, dim1) ? getDiagValue(a, idx, dim1) : zero(T)
)

isOnDiag(a::AbstractDiagonalMatrix, i::Int, diagLength::Int) = (i - 1) % (diagLength + 1) == 0

isOnDiag(a::AbstractDiagonalArray{T, N}, I, dim1::Int) where {T, N} = (
    return !any(i -> I[i] != dim1, 2:N)
)

# ======== Concrete Types ===================================================================== #

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           DiagonalArray:                                      #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct DiagonalArray{T, N, diagonal <: NTuple{N, T}} <: AbstractDiagonalArray{T, N} end

const DiagonalMatrix{T, diagonal} = DiagonalArray{T, 2, diagonal}

# ======== Constructor ======================================================================== #

DiagonalMatrix(size::NTuple{2, Int}, diagonal::Vararg{T, M}) where {N, T, M} = (
    allequal(size)   || @throw_error ArgumentError ERR_IS_NOT_SQUARE;
    first(size) == M || @throw_error ArgumentError ERR_DIAGONAL_DIM;
    return DiagonalArray{T, 2, diagonal}();
)

DiagonalArray(size::NTuple{N, Int}, diagonal::Vararg{T, M}) where {N, T, M} = (
    allequal(size)   || @throw_error ArgumentError ERR_IS_NOT_SQUARE;
    first(size) == M || @throw_error ArgumentError ERR_DIAGONAL_DIM;
    return DiagonalArray{T, N, diagonal}();
)

# ======== Functions for all DiagonalArray ==================================================== #

diagonal_length(a::DiagonalArray{T, N, diagonal}) where {T, N, diagonal} = length(diagonal)

getDiagValue(a::DiagonalMatrix{T, diagonal}, i::Int, d::Int) where {T, diagonal} = (
    posOnDiagonal = i % d;
    diagonal[posOnDiagonal]
)

getDiagValue(a::DiagonalArray{T, N, diagonal}, I, dim1::Int) where {T, N, diagonal} = diagonal[dim1]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           IdentityArray:                                      #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct IdentityArray{T, N, diaglength} <: AbstractDiagonalArray{T, N} end

const IdentityMatrix{T, diaglength} = IdentityArray{T, 2, diaglength}

# ======== Constructor ======================================================================== #

IdentityMatrix(diaglength::Int, ::Type{T}) where T = (
    return IdentityArray{T, 2, diaglength}();
)

IdentityMatrix(size::NTuple{2, Int}, ::Type{T}) where T = (
    allequal(size)   || @throw_error ArgumentError ERR_IS_NOT_SQUARE;
    return IdentityArray{T, 2, first(size)}();
)

IdentityArray(size::NTuple{N, Int}, ::Type{T}) where {N, T} = (
    allequal(size)   || @throw_error ArgumentError ERR_IS_NOT_SQUARE;
    return DiagonalArray{T, N, first(size)}();
)


# ======== Functions for all DiagonalArray ==================================================== #

diagonal_length(a::IdentityArray{T, N, diaglength}) where {T, N, diaglength} = diaglength

getDiagValue(a::IdentityArray{T, N, diagonal}, I, ::Int) where {T, N, diagonal} = one(T)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           ZeroArray:                                          #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
struct ZeroArray{T, N, dims} <: AbstractNDArray{T, N} end

# ======== Constructor ======================================================================== #
zeros(::Type{T}, size::NTuple{N, Int}) where {T, N} = ZeroArray{T, N, size}()

# ======== Functions to implement AbstractNDArray ============================================= #

Base.axes(a::ZeroArray{T, N, dims}) where {T, N, dims}  = map(Base.OneTo, dims)
Base.parent(a::ZeroArray)             = a
Base.IndexStyle(::ZeroArray)          = IndexLinear()

# Internal functions:
_getElement(::ZeroArray{T, 0})       where T  = zero(T)
_setElement!(::ZeroArray{T, 0}, val) where T  = zero(T)

_getElement(a::ZeroArray{T, N},  i::Int)                    where {T, N} = zero(T)
_getElement(a::ZeroArray{T, N},  I::AllCartesianIndex{N})   where {T, N} = zero(T)
_getElement(a::ZeroArray{T, N},  indices::NTuple{N, Index}) where {T, N} = zero(T)

_setElement!(::ZeroArray, val, i::Int)                                     = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::ZeroArray{T, N}, val, I::AllCartesianIndex{N}) where {T, N} = @throw_error ArgumentError ERR_NOT_STORED_VALUE
_setElement!(::ZeroArray{T, N}, val, indices::NTuple{N, Any}) where {T, N} = @throw_error ArgumentError ERR_NOT_STORED_VALUE