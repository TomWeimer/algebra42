# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                        compare Benchmark                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

using Printf
using Statistics

# ──── configuration ───────────────────────────────────────────────────────────────────────────── #

BENCH_FOLDER = "benchmark"

# ──── Benchmark Type ──────────────────────────────────────────────────────────────────────────── #

@enum LibOrigin begin
    myLib = 1
    baseLib = 2
end

struct BenchmarkTest
    # Name of the benchmark test:
    testName::AbstractString

    # Name of the function used in the test:
    functionUsed::AbstractString

    # Fastest execution:
    minimum_time::Float64

    # Average execution time:
    median_time::Float64

    # Library used
    origin::LibOrigin

    # Test file in which the test was found
    from::AbstractString
end

struct BenchmarkDiff
    # Name of the benchmark test:
    from::AbstractString
    testName::AbstractString
    function_used::AbstractString

    my_time::Float64
    base_time::Float64

    Δt::Float64
    ratio::Float64
end

using Printf


function format_benchmark_line(func, mylib, base, dt, pct)
    # Set fixed widths for columns
    w_func = 40        # function + description
    w_time = 12        # MyLib, Base, Δt times
    w_pct  = 8         # percentage column

    func_str  = rpad(func, w_func)                   # left-align function name
    mylib_str = lpad(@sprintf("%.3e μs", mylib), w_time)
    base_str  = lpad(@sprintf("%.3e μs", base), w_time)
    dt_str    = lpad(@sprintf("%+.3e μs", dt), w_time)
    pct_str   = lpad(@sprintf("%.2f%%", pct), w_pct)

    return "$func_str | MyLib: $mylib_str | Base: $base_str | Δt = $dt_str | $pct_str 🐢 slower"
end

function Base.show(io::IO, log::BenchmarkDiff)
    reset = "\033[0m"
    faster = log.ratio < 0
    color = faster ? "\033[32m" : "\033[31m"
    icon  = faster ? "⚡ faster" : "🐢 slower"

    name = log.function_used * " on " * log.testName
    
    @printf(io, "%-35s | MyLib: %10.3e μs | Base: %10.3e μs | Δt = %+10.3e μs | %s%6.2f%%%s %s\n",
        name, log.my_time, log.base_time, log.Δt, color, abs(log.ratio), reset, icon)
end

# ═══════════════════════════════════════════ parsing ════════════════════════════════════════════ #

# We parse all benchmark files to obtian each benchmark test


function obtain_all_benchmark_files() 
    benchmark_files = []

    # We iterate through all files in benchmark folder
    for (root, dirs, files) in walkdir("./$BENCH_FOLDER")
        for file in files
            if endswith(file, ".log")
                full_path = joinpath(root, file)
                push!(benchmark_files, full_path)
            end
        end
    end
    return benchmark_files
end

function obtain_all_benchmark_tests(benchmark_files)
    all_benchmark_tests = Dict()

    for benchmark_file in benchmark_files
        from = basename(benchmark_file)
        origin = startswith("my_", from) ? myLib : baseLib 
        benchmark_tests = obtain_benchmark_tests(benchmark_file, from, origin)

        # merge properly: append to existing array if key exists
        for (testName, tests) in benchmark_tests
            append!(get!(all_benchmark_tests, testName, Vector{BenchmarkTest}()), tests)
        end
    end

    return all_benchmark_tests
end

function obtain_benchmark_tests(benchmark_file, from, origin)
    benchmark_tests = Dict()

    lines = open(benchmark_file, "r") do file
        readlines(file)
    end

    isBenchmarkTest = false

    i = 1
    # We then iterate on the lines
    while i <= length(lines)
        line = lines[i]
        if !isBenchmarkTest && occursin("Benchmark in", line)
            isBenchmarkTest = true
            benchmark_lines = (line, lines[i+1], lines[i+2], lines[i+3])
            benchmark_test = createBenchmarkTest(benchmark_lines, from, origin)
            push!(get!(benchmark_tests, benchmark_test.testName, BenchmarkTest[]), benchmark_test)
            i += 4
            isBenchmarkTest = false
        else
            i+=1
        end
    end
    return benchmark_tests
end

function createBenchmarkTest(benchmark_lines::NTuple{4, String}, from, origin)
    testName, function_used, minimum_time, median_time = nothing, nothing, nothing, nothing

    # Use regex to capture both groups
    m1 = match(r"Benchmark in (.*?) for (.*?):", benchmark_lines[1])

    if m1 !== nothing
        testName = m1.captures[1]
        function_used = m1.captures[2]
    end

    m2 = match(r"  minimum time: (.*?) μs", benchmark_lines[3])

    if m2 !== nothing
        minimum_time = parse(Float64, m2.captures[1])
    end

    m3 = match(r"  median time: (.*?) μs", benchmark_lines[4])

    if m3 !== nothing
        median_time = parse(Float64, m3.captures[1])
    end

    any(isnothing, [testName, function_used, minimum_time, median_time, from, origin]) && error(
        "The parsing of the benchmark file $from failed"
    )
    
    return BenchmarkTest(testName, function_used, minimum_time, median_time, origin, from)
end

function compare_bench_logs(all_benchmark_tests)
    all_benchmark_diffs = Dict{String, BenchmarkDiff}()

    for (testName, logs) in all_benchmark_tests
        if length(logs) != 2
            error("Expected 2 benchmark logs for $testName, got $(length(logs))")
        end
        log1, log2 = logs
        my_log, base_log = log1.origin == myLib ? (log1, log2) : (log2, log1)
        Δt = my_log.median_time - base_log.median_time
        ratio = my_log.median_time / base_log.median_time
        benchmark_diff = BenchmarkDiff(
            my_log.from, 
            testName, 
            my_log.functionUsed, 
            my_log.median_time, 
            base_log.median_time,
            Δt,
            ratio
        )
        push!(all_benchmark_diffs, testName => benchmark_diff)
    end
    return all_benchmark_diffs
end

function printHeader()
    reset = "\033[0m"
    yellow = "\e[33m"
    println("╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮")
    println("│   🔥                                        $yellow Benchmark Comparison Report    $reset                                      🔥   │")
    println("╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯\n")
end

# function heatbar(ratio)
#     ratio < 1 && return "\e[32m" * "█"^round(Int, (1-ratio)*10) * "\e[0m"
#     ratio > 1 && return "\e[31m" * "█"^round(Int, (ratio-1)*10) * "\e[0m"
#     return ""
# end

# function display_heat_summary(grouped::Dict{String, Vector{BenchmarkDiff}})
#     println("\n🔥 Benchmark Heatmap Summary 🔥")
#     for (func, diffs) in grouped
#         println("\n$func:")
#         for d in diffs
#             println("  ", rpad(d.testName, 25),
#                     @sprintf("%8.2fx", d.ratio), " ", heatbar(d.ratio))
#         end
#     end
# end

# Group BenchmarkDiff by function_used
function group_by_function(diffs::Dict{String, BenchmarkDiff})
    grouped = Dict{String, Vector{BenchmarkDiff}}()
    for (_, d) in diffs
        push!(get!(grouped, d.function_used, Vector{BenchmarkDiff}()), d)
    end
    return grouped
end


# function summarize_function_stats(grouped::Dict{String, Vector{BenchmarkDiff}})
#     summary = Dict{String, NamedTuple}()

#     for (func, diffs) in grouped
#         n = length(diffs)
#         total_Δt = sum(d.Δt for d in diffs)
#         avg_Δt = mean(d.Δt for d in diffs)
#         avg_ratio = mean(d.ratio for d in diffs)
#         median_ratio = median(d.ratio for d in diffs)

#         summary[func] = (
#             n = n,
#             total_Δt = total_Δt,
#             avg_Δt = avg_Δt,
#             avg_ratio = avg_ratio,
#             median_ratio = median_ratio,
#             faster = avg_ratio < 1
#         )
#     end

#     return summary
# end


# Return a colored bar representing performance ratio
function heatbar(ratio; maxlen=20)
    # Limit extreme values to make the bar readable
    capped_ratio = clamp(ratio, 0.5, 2.0)  # anything below 0.5 treated as 0.5, above 2.0 as 2.0

    if capped_ratio < 1
        n = round(Int, (1 - capped_ratio) / 0.5 * maxlen)  # scale to maxlen
        return "\e[32m" * "█"^n * "\e[0m"  # Green = faster
    elseif capped_ratio > 1
        n = round(Int, (capped_ratio - 1) / 1.0 * maxlen)  # scale to maxlen
        return "\e[31m" * "█"^n * "\e[0m"  # Red = slower
    else
        return " "  # Neutral
    end
end

# Summarize all diffs per function (average ratio, min, max)
# Summarize all diffs per function (average, min, max)
function summarize_function_stats(diffs::Vector{BenchmarkDiff})
    ratios = [d.ratio for d in diffs]
    return (
        avg = mean(ratios),
        min = minimum(ratios),
        max = maximum(ratios),
        count = length(ratios)
    )
end
# Display heatmap summary per function (aggregate)
function display_heat_summary_by_function(grouped::Dict{String, Vector{BenchmarkDiff}})
    println("\n🔥 Benchmark Heatmap Summary (by function) 🔥\n")

    # Sort functions alphabetically
    for func in sort(collect(keys(grouped)))
        diffs = grouped[func]
        stats = summarize_function_stats(diffs)

        # Format everything to fixed width for proper alignment
        func_str   = rpad(func, 20)
        avg_str    = @sprintf("%7.2fx", stats.avg)
        min_str    = @sprintf("%7.2fx", stats.min)
        max_str    = @sprintf("%7.2fx", stats.max)
        n_str      = @sprintf("%2d", stats.count)
        bar_str    = heatbar(stats.avg)

        println("Function: $func_str  | avg: $avg_str  min: $min_str  max: $max_str  n=$n_str $bar_str")
    end
end

function benchmarkAll()
    benchmark_files = obtain_all_benchmark_files()
    benchmark_tests = obtain_all_benchmark_tests(benchmark_files)
    benchmark_diffs = compare_bench_logs(benchmark_tests)

    printHeader()

    for (_, benchmark_diff) in benchmark_diffs
        println(benchmark_diff)
    end

    benchmark_by_functions = group_by_function(benchmark_diffs)
    display_heat_summary_by_function(benchmark_by_functions)
end

benchmarkAll()
exit()
