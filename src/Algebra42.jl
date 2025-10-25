
"""
    Algebra42

A simple library similar to numpy.
"""
module Algebra42

export Collection

const Collection = Base.Vector

using Infiltrator


# 1. no dep
include("math/BasicMath.jl")
include("container/PaddedShape.jl")
include("other/Errors.jl")
include("other/macro.jl")
include("other/constant.jl")
include("container/Shape.jl")

include("utils/indexUtils.jl")
include("utils/shapeUtils.jl")


include("iterator/CartesianIndex.jl")
include("iterator/CartesianIndices.jl")



#include("container/Specials.jl")

# 2.
include("container/NDArray/AbstractNDArray.jl")
include("container/views/NDSubArray.jl")
include("container/views/ReshapeView.jl")
include("container/NDArray/NDArray.jl")

include("container/Transpose.jl")


include("iterator/Iter.jl")
include("iterator/MultiIter.jl")
include("math/LinearMath.jl")



# Export the functions you want to make publicly available
export  ReshapedArray42, Shape, ndims, NDArray, OnesArray, ZerosArray, OnesVector, ZerosVector, SingleOneVector, length, size,  nindexes, MultiIter, Iter, add, sub, prod, Vector, Matrix, linear_combination, norm_1, norm, norm_inf, dot, lerp

end # module Algebra42
