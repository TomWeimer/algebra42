const Matrix{DType} = NDArray{DType, 2}

# scalar 
Matrix{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Matrix{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Matrix{DType}(data::AbstractArray{T, 2}) where {DType, T} = _init(data, Shape(size(data)), T)

# Uninitialized constructors from shape:
Matrix{DType}(shape::Tuple) where {DType} = NDArray{DType}(Shape(shape))

Matrix{DType}(shape::Shape{2}) where {DType} = NDArray( Array{DType, 2}(undef, shape.length), shape, compute_strides(shape) )

function zeroMatrix(shape::Tuple, type::Type{T}) where {T}
    rtn = Matrix{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end


function Base.hcat(A::Matrix{T}, B::Matrix{T}) where {T}
    
    size(A, 1) == size(B, 1) || throw(ArgumentError("Those matrix have not the same height"))

    shape = ( size(A, 1), size(A, 2) + size(B, 2) )

    dest = Matrix{T}(shape)

    m, n1 = size(A, 1), size(A, 2)
    
    for j in 1:shape[2]
        
        if (j <= n1)
            for i in 1:m
                dest[i, j] = A[i, j]
            end
        else
            for i in 1:m
                dest[i, j] = B[i, j - n1]
            end
        end
    end
    return dest
end

function Base.vcat(A::Matrix{T}, B::Matrix{T}) where {T}
    
    size(A, 2) == size(B, 2) || throw(ArgumentError("Those matrix have not the same height"))

    shape = ( size(A, 1) + size(B, 1), size(A, 2) )

    dest = Matrix{T}(shape)

    m1 = size(A, 1)
    
    for j in 1:shape[2]
        for i in 1:shape[1]
            dest[i, j] = (i <= m1) ? A[i, j] : B[i - m1, j]
        end
    end
    return dest
end


function isSquare(A::Matrix)
    return size(A, 1) == size(A, 2)
end

function identityMatrix(n, type::Type{T}) where T
    identity = Matrix{type}((n, n))

    fill!(identity, 0)

    for k in 1:n
        identity[k, k] = one(T)
    end
    return identity
end