# -------------------- Minimal Julia Test Runner --------------------

const TEST_FOLDER = "tests"

"Discover all Julia test files in TEST_FOLDER"
function discover_tests(test_folder::AbstractString)
    test_files = String[]
    for (root, _, files) in walkdir(test_folder)
        for f in files
            endswith(f, ".jl") && push!(test_files, joinpath(root, f))
        end
    end
    return test_files
end

"Run a single Julia test file"
function run_test_file(file::AbstractString)
    println("Running $file")
    try
        include(file)
        filename = basename(file)[1:end-3]
        fn_name = Symbol("run_tests_", filename)

        if @isdefined fn_name
            getfield(Main, fn_name)(println)   # run test function if defined
        else
            println("⚠️ No run_tests_$filename function found")
        end
        return true
    catch e
        @error "Test failed" file exception=(e, catch_backtrace())
        return false
    end
end

# -------------------- Entry point --------------------

files_to_run =
    if length(ARGS) > 0
        [ARGS[1]]   # run only the given file
    else
        if !isdir(TEST_FOLDER)
            println("ERROR: Test folder '$TEST_FOLDER' does not exist")
            exit(1)
        end
        discover_tests(TEST_FOLDER)
    end

all_passed = all(run_test_file(f) for f in files_to_run)

# -------------------- Summary --------------------

if all_passed
    println("✅ All Julia tests passed successfully!")
    exit(0)
else
    println("❌ Some Julia tests failed or errored!")
    exit(1)
end