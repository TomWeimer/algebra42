
"""
    Algebra42

A simple library similar to numpy.
"""
module Algebra42

using Base: @propagate_inbounds, @inbounds, @boundscheck

# 1. no dep
include("Math/BasicMath.jl")
include("Other/Errors.jl")
include("Other/macro.jl")
include("Other/constant.jl")
include("Views/PaddedShape.jl")

include("Indexing/utils.jl")

# 2.
include("Containers/AbstractNDArray.jl")


include("Indexing/cartesian.jl")
include("Broadcast/Broadcasted.jl")
include("Broadcast/MyBroadcast.jl")
include("Indexing/fancy.jl")
include("Views/NDSubArray.jl")


include("Views/PermutedDimsArray.jl")
include("Containers/NDArray.jl")
include("Views/ReshapeView.jl")


include("Views/Transpose.jl")
include("Containers/SpecialArray.jl")

include("Math/LinearMath.jl")



# Export the functions you want to make publicly available
export  ReshapedArray42,ndims, NDArray,  length, size, add, sub, prod, Vector, Matrix, 
linear_combination, norm_1, norm, norm_inf, dot, lerp, @MyBroadcast, 
materialize42, materialize42!, Broadcasted42

end # module Algebra42
