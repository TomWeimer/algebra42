import Pkg; 
Pkg.add("BenchmarkTools")

using BenchmarkTools

N = 10^7

# Vector of Bool (1 byte per element)
boolvec = Vector{Bool}(rand(Bool, N))
# BitArray (1 bit per element)
bitarr = BitArray(rand(Bool, N))

@btime sum($boolvec)   # Fast
@btime sum($bitarr)    # ~2–3× slower