using Pkg, Test, Dates, Printf, Infiltrator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                              Tests:                                              #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# ════════════════════════════════════════ prepare tests ═════════════════════════════════════════ #

# ──── constants ───────────────────────────────────────────────────────────────────────────────── #

# Folders
const TEST_FOLDER  = "tests/"
const TEST_PREFIX  = "test"
const LOG_FOLDER   = "logs/"
const BENCH_FOLDER = "benchmark/"

# Files
const TEST_DATA    = "./data/TestData.jl"
const TEST_UTILS   = "./data/TestUtils.jl"
const DATA_LOGFILE = "data/julia_data.log"

# Other
const TIMESTAMP  = length(ARGS) > 0 ? ARGS[1] : Dates.format(Dates.now(), "yyyyMMdd_HHMMSS")
const CREATED_AT = "created at: [$TIMESTAMP]"
const PROJECT_PATH = joinpath(@__DIR__, "..")

# Note: not all of these are used.
const PACKAGES = [
    "Infiltrator", 
    "Debugger", 
    "Profile", 
    "ProfileView", 
    "BenchmarkTools", 
    "GeometryBasics",
    "LinearAlgebra",
    "RowEchelon",
    "Printf",
    "Statistics"
]

const RUN_BENCHMARK = length(ARGS) > 1 && ARGS[2] == "--benchmark"



# ──── variables ───────────────────────────────────────────────────────────────────────────────── #

BENCHMARK_FILE = ""
LOG_FILE       = joinpath(LOG_FOLDER, DATA_LOGFILE)

# ──── file and folder ─────────────────────────────────────────────────────────────────────────── #

"Ensure all folder exists"
ensure_folder(paths::AbstractString...) = foreach(ensure_folder, paths)

"Ensure a folder exists"
ensure_folder(file_path::AbstractString) = isdir(dirname(file_path)) || mkpath(dirname(file_path))

"Create a file with timestamp"
function create_file(full_path, to_replace...)
    file_path = replace(full_path, to_replace...)
    ensure_folder(file_path)
    open(file_path, "w") do file
        println(file, CREATED_AT)
    end
    return file_path
end

"Create a logfile"
logfile(file_path)  = create_file(file_path)

"Create a logfile from test file"
logfile(root, file)  = create_file(
    joinpath(root, file),
    TEST_FOLDER => LOG_FOLDER, 
    ".jl"  => ".log"
)

"Create a filepath for benchfiles"
benchfile(root, file) =  create_file(
    joinpath(root, file),
    TEST_FOLDER => BENCH_FOLDER, 
    ".jl"  => ".log"
)

"Create the files needed for the tests"
function create_files()
    test_files, log_files, bench_files = String[], String[], String[]

    for (root, _ , files) in walkdir(TEST_FOLDER)
        for file in files
            if endswith(file, ".jl")
                full_path = joinpath(root, file)
                push!(test_files, full_path)
                push!(log_files,  logfile(root, file))
                push!(bench_files, benchfile(root, file))
            end
        end
    end

    return test_files, log_files, bench_files
end

"return the filename of the path"
filename(path::AbstractString) = basename(path)[1:end-3]

# ──── log functions ───────────────────────────────────────────────────────────────────────────── #

"Change log file"
setLogFile(file) = global LOG_FILE = file

"Write a message to a given logfile"
write_log(msg::AbstractString) =
    open(LOG_FILE, "a") do f
        println(f, "$msg")
    end

"Log data content for comparing Julia and Numpy behavior"
function logDataContent()

    logfile(LOG_FILE) # create the first logfile
    
    # Log all test data content
    for (name, data) in TestData.allData
        write_log("\n" * name)
        write_content(data, write_log)
    end
end

# ──── benchmark functions ─────────────────────────────────────────────────────────────────────── #

using BenchmarkTools

"change the benchmark file"
setBenchmarkFile(file) = global BENCHMARK_FILE = file

write_bench(msg::AbstractString) =
    open(BENCHMARK_FILE, "a") do f
        println(f, "$msg")
    end

"Runs `fn(args...)` under BenchmarkTools, logs the result, and returns the BenchmarkTools.Trial object."
function run_benchmark(testname::AbstractString, fn::Function; args=(), function_used = nameof(fn))
    RUN_BENCHMARK || return nothing 
    # Run the benchmark
    trial = @benchmark $(() -> fn(args...))()  # interpolate args to avoid timing overhead

    # Get summary as string
    summary_str = "Benchmark in $testname for $function_used:\n" *
                  "  samples: $(length(trial))\n" *
                  "  minimum time: $(minimum(trial.times)/1e6) μs\n" *
                  "  median time: $(median(trial.times)/1e6) μs\n"
    # Log the result
    write_bench(summary_str)

    return trial
end

# ──── in/out ──────────────────────────────────────────────────────────────────────────────────── #

"execute a function in silent mode"
function quiet(f::Function, args...) 
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            f(args...)
        end
    end
end

"change the ouput files to the next ones"
function set_output_files(log_files, bench_files, i)
    setLogFile(log_files[i])
    setBenchmarkFile(bench_files[i])
end

"Ensure that test folder exist"
check_test() = isdir(TEST_FOLDER) ?  nothing : error("The test folder $TEST_FOLDER do not exist")


function print_progress(i, n, file; width=30)
    # Percentage
    pct = i / n
    nfilled = round(Int, pct * width)
    nempty  = width - nfilled
    bar = "█"^nfilled * "-"^nempty
    # Truncate file name if too long
    fname = basename(file)
    if length(fname) > 20
        fname = "..." * fname[end-16:end]
    end
    @printf("\r[%s] %3d/%d %s ...", bar, i, n, fname)
    flush(stdout)
end

"run the different testfiles"
function runtests(test_files, log_files, bench_files)
    n_tests = length(test_files)
    println("\n💡 Running tests ($n_tests total)")

    for (i, file) in enumerate(test_files)

        # We change the logfile and benchmark file
        set_output_files(log_files, bench_files, i)
        print_progress(i, n_tests, file)
        
        # Determine the entry function name for the test
        fn_name = Symbol("run_tests_", filename(file))

        try
            # We include the test file
            include(file)

            # And run the test
            if @isdefined(fn_name)
                eval(:( $(fn_name)(write_log, run_benchmark) ))
            else
                println("Function $fn_name not found in $file")
                exit(1)
            end
        catch e
           if isa(e, SystemError) && occursin("No such file or directory", e.msg)
                println("File not found: $file")
                exit(1)
            else
                println("An error happened in $file")
                rethrow(e)
            end
        end
    end
end


# ══════════════════════════════════════════ run tests ═══════════════════════════════════════════ #

# We start to check that the tests are here:

check_test()

# Then we are making sure that all folders are created.

ensure_folder(TEST_FOLDER, LOG_FOLDER, BENCH_FOLDER)

# We then activate the project

quiet(Pkg.activate, PROJECT_PATH)

# We add the necessary Packages

quiet(Pkg.add, PACKAGES)


# We include test functions and datas

include(TEST_DATA)
include(TEST_UTILS)

using .TestData, .TestUtils

# Log the basics test data that will be used for simple tests

logDataContent()

# We create all files needed for this project

test_files, log_files, bench_files  = create_files()

# We now need to run the tests
runtests(test_files, log_files, bench_files)


println("Julia tests finished")

exit(0)