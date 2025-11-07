using .Errors: ERR_DIM_HEIGHT, ERR_DIM_WIDTH

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           NDArray:                                            #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

mutable struct NDArray{DType,N} <: AbstractNDArray{DType,N}
   content::Union{Array{DType,1},Array{DType,0}}
   shape::Shape{N}
   strides::Tuple{Vararg{Int}}

   # Default constructor
   NDArray(content::AbstractArray{T}, shape::Shape{N}, strides) where {T,N} =
        new{T, N}(content, shape, strides)
end


const NestedArray{T} = AbstractArray{T, 1}

# ======== Constructors ======================================================================= #

# Scalar
NDArray{T}(x::Number) where {T} = (
   _init(x, T)
)

# From multidimensional array
NDArray{T}(data::AbstractArray{U, N}) where {T, U, N} = (
    _init(data, Shape(size(data)), T)
)

# Ragged nested array
NDArray{Any}(data::NestedArray{T}) where {T} = (
    _init(data, Shape(data; dtype=Any), Any)
)

# Nested array (Numpy format)
NDArray{T}(data::NestedArray{U}) where {T, U} = (
    _init(data, Shape(data), T)
)

# From shape (tuple)
NDArray{T}(shape::Tuple) where {T} = (
    NDArray(Array{T, 1}(undef, prod(shape)), Shape(shape), compute_strides(shape))
)

# ======== Constructors helpers ======================================================================= #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                            _init:                                             #
#_______________________________________________________________________________________________#

# Scalar
_init(x::Number, T::Type) = (
    _init(fill(T(x)), Shape(x))
)

# Array with explicit type and shape
_init(data::AbstractArray{U}, shape::Shape{N}, T::Type, strides=compute_strides(shape)) where {U, N} = (
    _init(_fill(data, shape, T), shape, strides)
)

# Generic catch-all
_init(content::AbstractArray{T}, shape::Shape{N}, strides=compute_strides(shape)) where {T, N} = (
    NDArray(content, shape, strides)
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                            _fill:                                             #
#_______________________________________________________________________________________________#

# Fill using multidimentional array
function _fill(data::AbstractArray{U,N}, shape::Shape{N}, dtype::Type{T}) where {U,N,T}
    content = Array{dtype, 1}(undef, shape.length)
    for (i, val) in enumerate(data)
      content[i] = dtype(val)
    end
   return content
end

# Fill using ragged array
function _fill(data::NestedArray{U}, shape::Shape{N}, dtype::Type{Any}) where {U,N}
    content = similar(data, dtype, shape.length)
    for i in eachindex(content)
        content[i] = data[i]
    end
    content
end

# Fill using nested array
function _fill(data::NestedArray{U}, shape::Shape{N}, dtype::Type{T}) where{U, N, T}
    content = similar(data, dtype, shape.length)
    sizeA = size(shape)

    # to change from how data is stored in nested arrays we need the following operations:
    flat     = flatten(data, sizeA, dtype)
    tmp      = reshape(flat, reverse(sizeA)...)
    colmajor = permutedims(tmp, reshaped_dims(sizeA))

    # then we copy the data in our linear container  
    for (i, idx) in enumerate(eachindex(colmajor))
        content[i] = colmajor[idx]
    end
    return content
end

# ======== Functions to implement AbstractNDArray ============================================= #

# Base functions:

Base.axes(a::NDArray)      = map(Base.OneTo, a.shape.dims)
Base.parent(a::NDArray)    = a
Base.IndexStyle(::NDArray) = IndexLinear()

# Internal functions:

# Scalar
@propagate_inbounds _getElement(a::NDArray{T, 0})          where {T} = a.content[1]
@propagate_inbounds _setElement!(a::NDArray{T, 0}, val::T) where {T} = a.content[1] = val

# Linear index:
@propagate_inbounds _getElement(a::NDArray{T, N}, i::Int)          where {T, N} = a.content[i]
@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, i::Int) where {T, N} = a.content[i] = val

# Cartesian index:
@propagate_inbounds _getElement(a::NDArray{T, N}, I::AllCartesianIndex{N})  where {T, N} = (
    a.content[ from_cartesian_to_linear(a, I) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, I::AllCartesianIndex{N}) where {T, N} = (
    a.content[ from_cartesian_to_linear(a, I) ] = val
)

# 'Normal' index:
@propagate_inbounds _getElement(a::NDArray{T, N}, indices::NTuple{N, Index})          where {T, N} = (
    a.content[ from_cartesian_to_linear(a, indices) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, indices::NTuple{N, Index}) where {T, N} = (
    a.content[ from_cartesian_to_linear(a, indices) ] = val
)

# ======== Functions for all NDArray ================================================== #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       Core properties:                                        #
#_______________________________________________________________________________________________#

Base.size(a::NDArray)    = a.shape.dims
Base.strides(a::NDArray) = a.strides

Base.stride(a::NDArray, k::Integer) = ( 
    @boundscheck checkindex(Bool, axes(a, k), k); 
    @inbounds a.strides[k]
)

Base.pointer(A::NDArray) = pointer(A.content)
flatten(array::NDArray)  = array.content

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       similar functions:                                      #
#_______________________________________________________________________________________________#

# Similar: (create array of same size but uninitialized)
Base.similar(A::AbstractNDArray{T}) where {T} = NDArray{T}(size(A))

Base.similar(A::AbstractNDArray, ::Type{T}) where {T} = NDArray{T}(size(A))

Base.similar(::AbstractNDArray{T}, dims::Tuple{Vararg{Int}}) where {T} = NDArray{T}(dims)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                               internal helpers for NestedArray                                #
#_______________________________________________________________________________________________#

# flatten the nested array
function flatten(data::NestedArray{U}, sizeA::NTuple{N, Int}, dtype::Type) where {U, N} 
    flattenData = Array{dtype, 1}(undef, prod(sizeA))

    for (i, I) in enumerate(CartesianIndices_42(sizeA, order=RowOrder))
        flattenData[i] = dtype(_get_nested(data, I))
    end
    return flattenData
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

# Returns the dimension to permute to obtain col major data
reshaped_dims(::Tuple{Int}) = (1,)
reshaped_dims(::Tuple{Int, Int}) = (2, 1)

function reshaped_dims(::NTuple{N, Int}) where N
    trailing = ntuple(i -> N - i + 1, N - 2)
    return (2, 1, trailing...)
end

# ======== NDArray Aliases ============================================================== #

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                           Vector:                                             #
#_______________________________________________________________________________________________#

const Vector{DType} = NDArray{DType, 1}

# scalar 
Vector{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Vector{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Vector{DType}(data::AbstractArray{T, 1}) where {DType, T} = _init(data, Shape(size(data)), T)

# Uninitialized constructors from shape:

Vector{DType}(shape::Tuple{Int}) where {DType} = NDArray{DType}(shape)

function zeroVector(shape::Tuple, type::Type{T}) where {T}
    rtn = Vector{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                           Matrix:                                             #
#_______________________________________________________________________________________________#

const Matrix{DType} = NDArray{DType, 2}

# scalar 
Matrix{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Matrix{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Matrix{DType}(data::AbstractArray{T, 2}) where {DType, T} = _init(data, Shape(size(data)), DType)

# Uninitialized constructors from shape:
Matrix{DType}(shape::Tuple) where {DType} = NDArray{DType}(shape)

# ======== Functions for all Matrix ========================================================== #


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                     Concatenation:                                            #
#_______________________________________________________________________________________________#

function Base.hcat(A::AbstractNDMatrix{T}, B::AbstractNDMatrix{T}) where {T}
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

function Base.vcat(A::AbstractNDMatrix{T}, B::AbstractNDMatrix{T}) where {T}
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