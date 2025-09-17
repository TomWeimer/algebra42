using Pkg
Pkg.add("BenchmarkTools")

using BenchmarkTools

# Create a large mixed tuple with random numbers and strings
mixed_tuple = tuple(rand(1:100, 50_000)..., rand(["a", "b", "c"], 50_000)...)

# Create a large tuple with only numbers
numbers_only_tuple = tuple(rand(1:100, 100_000)...)

# Benchmark 1: Using a generator with a conditional filter
println("Benchmarking with a generator:")
@btime maximum(x for x in $mixed_tuple if x isa Number)

println("\n" * "-"^30 * "\n")

# Benchmark 2: Using the pre-filtered variable
println("Benchmarking with a pre-filtered variable:")
@btime maximum($numbers_only_tuple)