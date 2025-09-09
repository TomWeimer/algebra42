
"""
    Algebra42

A simple library similar to numpy.
"""
module Algebra42

export Collection

const Collection = Vector

using Infiltrator

include("MyException.jl")
include("Specials.jl")
include("MyMath.jl")
include("Shape.jl")
include("NDArray.jl")


# Export the functions you want to make publicly available
export prod, Shape, ndims, NDArray, OnesArray, ZerosArray, OnesVector, ZerosVector, SingleOneVector, length, size, item, nindexes

end # module Algebra42
