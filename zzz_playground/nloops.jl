using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools

using Base.Cartesian

function sum_with_sum_function(A::AbstractArray)
    return sum(A)
end

function sum_with_linear_indexing(A::AbstractArray)
    total_sum = 0.0
    for i in eachindex(A)
        @inbounds total_sum += A[i]
    end
    return total_sum
end

# A function using CartesianIndices
function sum_with_cartesian(A::AbstractArray)
    total_sum = 0.0
    for I in CartesianIndices(A)
        @inbounds total_sum += A[I]
    end
    return total_sum
end

# A function using @nloops
@generated function sum_with_nloops(A::AbstractArray{T, N}) where {T, N}
    quote
        sum_val = zero($T)
        @nloops $N i d->(1:size(A, d)) begin
            @inbounds sum_val += @ncall $N getindex A i
        end
        return sum_val
    end
end

# Create a large N-dimensional array
my_array = rand(100, 100, 100)

println("Benchmarking with sum function:")
@btime sum_with_cartesian($my_array)

println("\n" * "-"^30 * "\n")

println("Benchmarking with LinearIndices:")
@btime sum_with_nloops($my_array)

println("\n" * "-"^30 * "\n")


println("Benchmarking with CartesianIndices:")
@btime sum_with_cartesian($my_array)

println("\n" * "-"^30 * "\n")

println("Benchmarking with @nloops:")
@btime sum_with_nloops($my_array)