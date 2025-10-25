include("NDArrayView.jl")
include("NDArray.jl")


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Strides:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Return the strides
Base.strides(a::NDArray)     = a.strides
Base.strides(a::NDArrayView) = a.strides

# Return the stride of along one axis
Base.stride(a::NDArray, k::Integer)     = ( @boundscheck checkindex(Bool, a, k); @inbounds a.strides[k])
Base.stride(a::NDArrayView, k::Integer) = ( @boundscheck checkindex(Bool, a, k); @inbounds a.strides[k])

# Compute Column Major strides from shape
function compute_strides(dims::Tuple)
    n = length(dims)
    strds = Vector{Int}(undef, max(n, 1))
    strds[1] = 1
    @inbounds for i in 2:n
        strds[i] = strds[i-1] * dims[i-1]
    end
    return Tuple(strds)
end

compute_strides(shape::Shape) = compute_strides(shape.dims)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Offset:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Use multiple indices to compute their corresponding offset (linear index) 

function offset(a::NDArrayView{T, N}, indices::Union{CartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing) where {T, N}
    return _offset(a, indices; default_offset = a.offset)
end

function offset(a::AbstractNDArray{T, N}, indices::Union{CartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing) where {T, N}
    return _offset(a, indices)
end

# Scalar
_offset(::AbstractNDArray{T, 0}, ::Nothing; default_offset::Int = 0) where T = default_offset + 1

# CartesianIndex
_offset(a::AbstractNDArray{T, N}, I::CartesianIndex{N}; default_offset::Int = 0) where {T, N} = (
    _offset(strides(a), Tuple(I); default_offset = default_offset)
)

# 'Normal' indices
_offset(a::AbstractNDArray{T, N}, indices::NTuple{N, Int}; default_offset::Int = 0) where {T, N} = (
    _offset(strides(a), indices; default_offset = default_offset)
)

# Metadata
function _offset(strides::NTuple{N, Int}, indices::NTuple{N, Int}; default_offset::Int = 0) where {N}
   offset = default_offset
   for i in 1:N
      offset += (indices[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1
end


