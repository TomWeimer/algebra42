using Pkg, Test, Dates, Printf, Algebra42, Infiltrator

# -------------------- Constants --------------------

const TEST_FOLDER = "tests"
const LOG_FOLDER = "logs"
const TEST_DATA = "./data/TestData.jl"
const TEST_UTILS = "./data/TestUtils.jl"
DEFAULT_LOG_FILE = joinpath(LOG_FOLDER, "data/julia_data.log")

# -------------------- Functions --------------------

"Return a timestamp string from ARGS or current time"
function obtainTimeStamp()
    return length(ARGS) > 0 ? ARGS[1] : Dates.format(Dates.now(), "yyyyMMdd_HHMMSS")
end

"Write a timestamped message to a given logfile"
function write_log(msg::AbstractString)
    open(DEFAULT_LOG_FILE, "a") do f
        println(f, "$msg")
    end
end

"Ensure a folder exists"
function ensure_folder(path::AbstractString)
    isdir(path) || mkpath(path)
end

"Discover all Julia test files in TEST_FOLDER and return their paths and corresponding log paths"
function discover_tests(test_folder::AbstractString, log_folder::AbstractString)
    test_files, test_logs = String[], String[]
    
    for (root, dirs, files) in walkdir(test_folder)
        for f in files
            if endswith(f, ".jl")
                full_path = joinpath(root, f)
                push!(test_files, full_path)
                
                # Relative path without TEST_FOLDER prefix
                relative_path = joinpath(splitpath(full_path)[2:end]...)
                relative_path_no_ext = replace(relative_path, ".jl" => ".log")
                log_path = joinpath(log_folder, relative_path_no_ext)
                
                ensure_folder(dirname(log_path))
                push!(test_logs, log_path)
            end
        end
    end
    
    return test_files, test_logs
end


# -------------------- Setup --------------------

# Timestamp
const RUN_TIMESTAMP = obtainTimeStamp()

# Ensure log folder exists
ensure_folder(LOG_FOLDER)
ensure_folder(dirname(DEFAULT_LOG_FILE))

# Get script directory
script_dir = @__DIR__

# Activate parent folder quietly
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Pkg.activate(joinpath(script_dir, ".."))
    end
end

# Install Infiltrator quietly
redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Pkg.add("Infiltrator")
    end
end

redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        Pkg.add("Debugger")
    end
end

# -------------------- Include test data and utils --------------------

include(TEST_DATA)
include(TEST_UTILS)
using .TestData, .TestUtils

write_log("created at: [$RUN_TIMESTAMP]")
# Log all test data content
for (name, data) in allData
    write_log("\n" * name)
    write_content(data, log -> write_log(log))
end

# -------------------- Discover tests --------------------

if !isdir(TEST_FOLDER)
    write_log("ERROR: Test folder '$TEST_FOLDER' does not exist")
    exit(1)
end

test_files, test_logs = discover_tests(TEST_FOLDER, LOG_FOLDER)

# -------------------- Run tests --------------------

all_passed = true

for (i, file) in enumerate(test_files)
    logfile = test_logs[i]
    println("Running $file -> $logfile")
    global DEFAULT_LOG_FILE = logfile
    write_log("created at: [$RUN_TIMESTAMP]")
    

        filename = basename(file)[1:end-3]
        fn_name = Symbol("run_tests_", filename)
       

        include(file)

        if @isdefined fn_name
            getfield(Main, fn_name)(write_log)   # pass println as write_log
        end

end

# -------------------- Final summary --------------------

if all_passed
    println("All Julia tests passed successfully!")
    exit(0)
else
    println("Some Julia tests failed or errored!")
    exit(1)
end