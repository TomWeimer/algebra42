
"""
    Algebra42

A simple library similar to numpy.
"""
module Algebra42

export Collection

const Collection = Base.Vector

using Infiltrator



include("math/BasicMath.jl")
include("other/MyException.jl")
include("container/Specials.jl")
include("container/Shape.jl")
include("container/NDArray/NDArray.jl")
include("container/Matrix/Matrix.jl")
include("container/Vector/Vector.jl")
include("container/NDArray/indexing.jl")
include("container/NDArray/dispatcher.jl")
include("iterator/CartesianIndex.jl")
include("iterator/CartesianIndices.jl")
include("container/NDArray/fancy.jl")
include("container/NDArray/NDArrayView.jl")
include("container/TransAdj.jl")
include("container/NDArray/operator.jl")
include("iterator/Iter.jl")
include("iterator/MultiIter.jl")
include("math/LinearMath.jl")


# Export the functions you want to make publicly available
export  Shape, ndims, NDArray, OnesArray, ZerosArray, OnesVector, ZerosVector, SingleOneVector, length, size, item, nindexes, MultiIter, Iter, add, sub, prod, Vector, Matrix, linear_combination, norm_1, norm, norm_inf, dot, lerp

end # module Algebra42
