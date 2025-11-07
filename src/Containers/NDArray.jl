using .Errors: ERR_DIM_HEIGHT, ERR_DIM_WIDTH

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                             NDArray:                                             #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

mutable struct NDArray{T,N} <: AbstractNDArray{T,N}
   content::Union{Array{T,1}, Array{T,0}}
   shape::Dims{N}
   strides::Tuple{Vararg{Int}}


   # Default constructor
   function NDArray(content::AbstractArray{T}, shape::Dims{N}, strides) where {T,N}
        any(d -> d < 0, shape) && throw(DomainError("Dimensions cannot be negative"))
        new{T, N}(content, shape, strides)
    end
end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #
            
const ScalarNDArray{T} = NDArray{T, 0}
const Vector{T}        = NDArray{T, 1}
const Matrix{T}        = NDArray{T, 2}

# ═════════════════════════════════════════ constructors ═════════════════════════════════════════ #`

# Scalar
NDArray{T}(x::Number) where T = (
   NDArray(fill(T(x)), (), ())
)

# Numpy format
NDArray{T}(data::NestedArray{U}) where {T, U} = (
    _init(data, getShape(data), T)
)

# Julia format
NDArray{T}(data::MultiDimArray{S, N}) where {T, S, N} = (
    _init(data, size(data), T)
)

# Ragged Array
NDArray{Any}(data::NestedArray{T}) where T = (
    _init(data, (length(data),), Any)
)

# Uninitialized from shape
NDArray{T}(shape::Dims) where T = (
    NDArray(Array{T, 1}(undef, prod(shape)), shape, compute_strides(shape))
)

# ──── constructor helper: init ────────────────────────────────────────────────────────────────── #

_init(data::NestedArray, row_shape::Dims{N}, T::Type) where N = (
    col_shape = (N <= 2) ? row_shape : permute(row_shape, reshaped_dims(row_shape));
    strides = compute_strides(col_shape);
    content = _fill(data, row_shape, col_shape, T);
    NDArray(content, col_shape, strides);
)

_init(data::MultiDimArray, shape::Dims, T::Type) = (
    strides = compute_strides(shape);
    NDArray(_fill(data, shape, T), shape, strides)
)

# ──── constructor helper: getShape ────────────────────────────────────────────────────────────── #

# Return the shape of nested arrays
function getShape(data::NestedArray)
    return isempty(data) ? (0,) : _getShape(data)
end

function _getShape(data)
    if isa(data, AbstractVector)
        subarrayDims = map(_getShape, data)
        
        # We have the following possibilites:
        if all(x -> isnothing(x), subarrayDims)
        # 1. The actual array don't contains array 
            return (length(data),);
        elseif allequal(subarrayDims)
        # 2. The actual array contains arrays but all of them have the same size
            return (length(data), subarrayDims[1]...);
        else
        # 3. The actual array contains arrays but they are not of the same size
            throw(DomainError("Cannot be a ragged array if type is not Any"))
        end
    else
        return nothing;
    end
end

# ──── constructor helper: fill ────────────────────────────────────────────────────────────────── #

function _fill(data::MultiDimArray{S,N}, shape::Dims, ::Type{T}) where {S,N,T}
    content = Array{T, 1}(undef, prod(shape))
    for (i, val) in enumerate(data)
      content[i] = T(val)
    end
   return content
end

function _fill(data::NestedArray, row_shape::Dims{N}, col_shape::Dims{N}, ::Type{T}) where {T, N}
    content = similar(data, T, (prod(col_shape),))

    # Tis flatten function is designed to return flatten data from row major in col major
    flat = flatten(data, T, row_shape)

    # We need to change the shape as the numpy format is for example:
    # (block, row, col) in numpy but (row, col, block) in julia (only for N > 2)
    if N <= 2
        colMajor = flat
    else
        colMajor = reshape(flat, col_shape...)
    end

    # then we copy the data in our linear container  
    for (i, idx) in enumerate(eachindex(colMajor))
        content[i] = colMajor[idx]
    end
    return content
end

# Ragged Array
function _fill(data::NestedArray, shape::Dims, ::Type{Any})
    content = similar(data, Any, prod(shape))
    for i in eachindex(content)
        content[i] = data[i]
    end
    content
end

# ══════════════════════════════════ implements AbstractNDArray ══════════════════════════════════ #
            
# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.axes(a::NDArray)      = map(Base.OneTo, a.shape)
Base.parent(a::NDArray)    = a
Base.IndexStyle(::NDArray) = IndexLinear()

# ──── Internal functions ──────────────────────────────────────────────────────────────────────── #

@propagate_inbounds _getElement(a::ScalarNDArray)       = a.content[1]
@propagate_inbounds _setElement!(a::ScalarNDArray, val) = a.content[1] = val

@propagate_inbounds _getElement(a::NDArray{T, N}, i::Index)       where {T, N} = a.content[i]
@propagate_inbounds _setElement!(a::NDArray{T, N}, val, i::Index) where {T, N} = a.content[i] = val


# Cartesian index:
@propagate_inbounds _getElement(a::NDArray{T, N}, I::AllCartesianIndex{N})  where {T, N} = (
    a.content[ LinearIndex(a, I) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val, I::AllCartesianIndex{N}) where {T, N} = (
    a.content[ LinearIndex(a, I) ] = val
)

# 'Normal' index:
@propagate_inbounds _getElement(a::NDArray{T, N}, indices::NTuple{N, Index}) where {T, N} = (
    a.content[ LinearIndex(a, indices) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val, indices::NTuple{N, Index}) where {T, N} = (
    a.content[ LinearIndex(a, indices) ] = val
)

# ═══════════════════════════════════ functions for all NDArray ══════════════════════════════════ #

# ──── core properties ─────────────────────────────────────────────────────────────────────────── #
            
Base.size(a::NDArray)    = a.shape
Base.strides(a::NDArray) = a.strides

Base.stride(a::NDArray, k::Integer) = ( 
    @boundscheck checkindex(Bool, axes(a, k), k); 
    @inbounds a.strides[k]
)

Base.pointer(A::NDArray) = pointer(A.content)

# ──── similar functions ───────────────────────────────────────────────────────────────────────── #
            
Base.similar(A::AbstractNDArray{T}) where T = NDArray{T}(size(A))

Base.similar(A::AbstractNDArray, ::Type{T}) where T = NDArray{T}(size(A))

Base.similar(::AbstractNDArray{T}, dims::Tuple{Vararg{Int}}) where T = NDArray{T}(dims)


# ══════════════════════════════════════ flatten functions ═══════════════════════════════════════ #

# ──── flatten colmajor ────────────────────────────────────────────────────────────────────────── #

# Return the underlying array
flatten(array::NDArray)  = array.content

# Create a copy of the flatten data
function Base.vec(a::AbstractNDArray)
    v = similar(a, eltype(a), (length(a), ))
    for (i, val) in enumerate(a)
        v[i] = val
    end
    return v
end

# ──── flatten rowmajor ────────────────────────────────────────────────────────────────────────── #


function flatten(data::NestedArray{U}, ::Type{T}, dims::Tuple{Int}) where {U, T}
    rtn = Array{T, 1}(undef, prod(dims))
    for i in 1:dims[1]
        rtn[i] = data[i]
    end
    return rtn
end

function flatten(data::NestedArray{U}, ::Type{T}, dims::Tuple{Int, Int}) where {U, T}
    rtn = Array{T, 1}(undef, prod(dims))
    idx = 1
    for c in 1:dims[2]
        for r in 1:dims[1]   
            rtn[idx] = T(data[r][c])
            idx += 1
        end
    end
    return rtn
end

function flatten(data::NestedArray{U}, ::Type{T}, dims::Tuple{Int, Int, Vararg{Int}}) where {U, T}
    result = T[]
    for i in 1:dims[1]
        append!(result, flatten(data[i], T, Base.tail(dims))...)
    end
    return result
end


# obtain the value in the nested array
function _get_nested(data, I::AllCartesianIndex)
    tmp = data
    indices = Tuple(I)
    for i in indices
        tmp = tmp[i]
    end
    return tmp
end

# In numpy it is (dim N, dim N - 1, ...., dim3, row, col)
# In julia it is (row, col, dim 3, ..., dim N - 1, dim N )
# We return the permutation to obtain julia shape from numpy format

# No permutation needed both are the same when N <= 2
reshaped_dims(::Tuple{Int}) = (1,)
reshaped_dims(::Tuple{Int, Int}) = (1, 2)

# Permutation are need:
function reshaped_dims(::NTuple{N, Int}) where N
    # Let's have an example the shape in numpy format is (5, 4, 3, 1, 2)
    # This means that dim N = 5, dim N - 1 = 4, dim N - 2 = 3, dim 1 = 1, dim 2 = 2
    # We want (dim1, dim2, ..., dim N) so we begin by cutting the last dimensions
    dim1, dim2 = (N - 1, N)
    # The remaining dim are (dim N, dims N - 1, ..., dim 3) we want (dim 3, ..., dim N - 1, dim N)
    # So the permutation for the remaining dims is:
    remaining_dims = ( N - i for i in 2:N-1 )

    # We assemble the final permutation:
    return (dim1, dim2, remaining_dims...)
end

# ════════════════════════════════════════════ Vector ════════════════════════════════════════════ #

# ──── constructors ────────────────────────────────────────────────────────────────────────────── #
            
Vector{T}(x::Number) where T = NDArray([T(x)], (1,), (1,))

Vector{T}(data::MultiDimArray{S, 1}) where {T, S} =
    NDArray{T}(data) 

Vector{T}(shape::Dims{1}) where T = NDArray{T}(shape)

# ──── vector functions ────────────────────────────────────────────────────────────────────────── #
            
function zeroVector(shape::Dims, type::Type{T}) where T
    rtn = Vector{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end

# ════════════════════════════════════════════ Matrix ════════════════════════════════════════════ #

# ──── constructors ────────────────────────────────────────────────────────────────────────────── #

Matrix{T}(x::Number) where T  = NDArray([T(x)], (1,1), (1,1))

Matrix{T}(data::MultiDimArray{S, 2}) where {T, S} =
    NDArray{T}(data)

Matrix{T}(shape::Dims{2}) where T = NDArray{T}(shape)

# ──── matrix functions ────────────────────────────────────────────────────────────────────────── #

function Base.hcat(A::AbstractNDMatrix{T}, B::AbstractNDMatrix{T}) where T
    rowsA, colsA = size(A)
    rowsB, colsB = size(B)    
    rowsA == rowsB || @throw_error ArgumentError ERR_DIM_HEIGHT
    
    dest  = Matrix{T}((rowsA, colsA + colsB ))

    # Fill A
    for j in 1:colsA, i in 1:rowsA
        dest[i, j] = A[i, j]
    end

    # Fill B
    for j in 1:colsB, i in 1:rowsB
        dest[i, colsA + j] = B[i, j]
    end

    return dest
end

function Base.vcat(A::AbstractNDMatrix{T}, B::AbstractNDMatrix{T}) where T
    rowsA, colsA = size(A)
    rowsB, colsB = size(B)
    colsA == colsB || @throw_error ArgumentError ERR_DIM_WIDTH

    dest = Matrix{T}((rowsA + rowsB, colsA))


    # Fill A
    for j in 1:colsA, i in 1:rowsA
        dest[i, j] = A[i, j]
    end

    # Fill B
    for j in 1:colsB, i in 1:rowsB
        dest[rowsA + i, j] = B[i, j]
    end

    return dest
end