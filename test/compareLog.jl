LOG_FOLDER = "logs"
DIFF_FOLDER = "diffs"


function compare_logs_ignore_patterns(test_log::String, ref_log::String, ignore::Regex)::Bool
    if !isfile(test_log) || !isfile(ref_log)
        return false
    end
    lines1 = split(read(test_log, String), '\n')
    lines2 = split(read(ref_log, String), '\n')

    if length(lines1) != length(lines2)
        return false
    end

    for (l1, l2) in zip(lines1, lines2)
        if replace(l1, ignore => "") != replace(l2, ignore => "")
            return false
        end
    end
    return true
end

function log_contains(test_log::String, patterns::Vector{String})::Bool
    content = read(test_log, String)
    all(occursin(pat, content) for pat in patterns)
end

function testAllLogs()
    for (root, dirs, files) in walkdir("./$LOG_FOLDER")
        logs = []
        testName = basename(root)
        for f in files

            if endswith(f, ".log")
                full_log_path = joinpath(root, f)   # define full_test_path
                push!(logs, full_log_path)
            end
        end
        compareAllLogs(testName, logs)
    end
end

function compareAllLogs(testName, logs)
    myfile = ""
    passed = []

    if (testName != "data")
        for file in logs
            if (contains(file, "my_"))
                myfile = file
            end
        end
    else
        myfile = logs[1]
    end
    
    for file in logs
        if (file != myfile)
            push!(passed, compare_logs(myfile, file))
        end
    end
    test(testName, all(passed))
end

function compare_logs(test_log::String, ref_log::String)::Bool
    if !isfile(test_log) || !isfile(ref_log)
        return false
    end

    log_content = read(test_log, String)
    ref_content = read(ref_log, String)

    if log_content == ref_content
        return true
    else
        write_diff(test_log, ref_log)
        return false
    end
end

function test(name::AbstractString, result::Union{Bool,Exception})
    # ANSI escape codes for colors
    green = "\e[32m"
    red = "\e[31m"
    yellow = "\e[33m"
    bold = "\e[1m"
    reset = "\e[0m"

    try
        if result === true
            println("   $green ✅  PASS$reset  | $bold$name$reset")
        elseif result === false
            println("   $red ❌  FAIL$reset  | $bold$name$reset")
        elseif result isa Exception
            println("   $yellow ⚠️  ERROR$reset | $bold$name$reset")
            println("      → $red$(typeof(result))$reset: $(result)")
        else
            println("   $yellow ❓ UNKNOWN$reset | $bold$name$reset")
        end
    catch e
        println("   $yellow ⚠️  ERROR$reset | $bold$name$reset")
        println("      → $red$(typeof(e))$reset: $(e)")
    end
end

function write_diff(test_log::String, ref_log::String)
    diff_file = replace(test_log, ".log" => ".diff")
    diff_file = replace(diff_file, "$LOG_FOLDER" => "$DIFF_FOLDER")

    # Make sure the parent folder exists
    mkpath(dirname(diff_file))

    try
        # Run diff and write output to diff_file
        run(pipeline(`diff -u $ref_log $test_log`, diff_file))
    catch e
        # diff exits with nonzero when files differ → ignore
    end
    println("      📝 Diff saved to $diff_file")
end


testAllLogs()
exit()