const Vector{DType} = NDArray{DType, 1}

# scalar 
Vector{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Vector{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Vector{DType}(data::AbstractArray{T, 1}) where {DType, T} = _init(data, Shape(size(data)), T)

# Uninitialized constructors from shape:

Vector{DType}(shape::Tuple{Int}) where {DType} = NDArray{DType}(Shape(shape))

Vector{DType}(shape::Shape{1}) where {DType} = NDArray( Array{DType}(undef, shape.length), shape, _compute_strides(shape) )

function zeroVector(shape::Tuple, type::Type{T}) where {T}
    rtn = Vector{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end