
# SingleValueArray
abstract type SingleValueArray{Value, N} <: AbstractArray{typeof(Value), N} end

Base.size(array::SingleValueArray) = array.dims

Base.getindex(array::SingleValueArray{Value}, indices::Vararg{Int}) where {Value} = Value


# OnesArray
struct OnesArray{Value, N} <: SingleValueArray{Value, N}
    dims::NTuple{N, Int}
end

OnesArray(size::NTuple{N, Int}) where {N} = OnesArray{1, N}(size) 

OnesArray{T}(size::NTuple{N, Int}) where {N, T} = OnesArray{one(T), N}(size)


# Zeros Array
struct ZerosArray{Value, N} <: SingleValueArray{Value, N}
    dims::NTuple{N, Int}
end

ZerosArray(size::NTuple{N, Int}) where {N} = ZerosArray{0, N}(size) 

ZerosArray{T}(size::NTuple{N, Int}) where {N, T} = ZerosArray{zero(T), N}(size)


# Ones Vector
struct OnesVector{Value} <: SingleValueArray{Value, 1}
    dims::NTuple{1, Int}
end


OnesVector(length::Int) = OnesVector((length,))

OnesVector(size::NTuple{1, Int}) = OnesArray{1, 1}(size) 

OnesVector{T}(size::NTuple{1, Int}) where {T} = OnesArray{one(T), 1}(size)


# ZerosVector
struct ZerosVector{Value} <: SingleValueArray{Value, 1}
    dims::NTuple{1, Int}
end

ZerosVector(size::NTuple{1, Int}) = ZerosArray{0, 1}(size) 

ZerosVector{T}(size::NTuple{1, Int}) where {T} = OnesArray{zero(T), 1}(size)

# The struct is parametric on the type T of the element
struct SingleOneVector{Value, N}  <: SingleValueArray{Value, N}
    index::Int
    length::Int
end

# Outer constructor for a specified type
 SingleOneVector(index::Int, length::Int)= SingleOneVector{Int}(index, length)

# Outer constructor for a specified type
 SingleOneVector{Value}(index::Int, length::Int) where {Value} = SingleOneVector{one(Value), 1}(index, length)

# Implement the required AbstractArray methods
Base.size(array::SingleOneVector{Value, N}) where {Value, N} = (array.length,)

Base.getindex(array::SingleOneVector{Value}, i::Int) where {Value} = (i == array.index) ? Value : zero(Value)


function nindexes(length, position, newValue)
    OnesVector(length) + (SingleOneVector(position, length) .* (newValue - 1))
end


function nindexes(length)
    OnesVector(length)
end