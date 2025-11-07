include("../other/constant.jl")

using Base: @propagate_inbounds

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                           LinearIndex                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# use strides from the array
@propagate_inbounds function LinearIndex(a::AbstractArray, I::Indices{N}) where N
    N == 0 && return 1

    strds = strides(a)
    offset = 0
    @inbounds for i in 1:N
        offset += (I[i] - 1) * strds[i]
    end
    return offset + 1
end


# compute column strides on the go
@propagate_inbounds function LinearIndex(dims::Dims, I::Indices{N}) where N
    N == 0 && return 1
    
    offset = 0
    stride = 1
    @inbounds for i in 1:N
        offset += (I[i] - 1) * stride
        stride *= dims[i]
    end
    return offset + 1
end