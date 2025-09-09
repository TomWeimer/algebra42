module TestUtils

export printTest, exceptionPassed

function printTest(name::AbstractString, result::Bool)
    try
        if result === true
            println("   ✅ Test passed: $name")
        else
            println("   ❌ Test failed: $name")
        end
    catch e
        println("   ❌ Test error: $name")
        println("      → $(typeof(e)): $(e.msg)")
    end
end

function printTest(name::AbstractString, f::Function, ExceptionType::Type)
    try
        result = f()
        println("   ❌ Test failed: $name")
        println("   → No exception thrown")
    catch e
        if isa(e, ExceptionType)
            println("   ✅ Test passed: $name")
        else
            println("   ❌ Test error: $name")
            println("      → $(typeof(e)): $(e.msg)")
        end
    end
end


function exceptionPassed(f::Function, nb::Int, ExceptionType::Type)
    passed = true;

    try
        x = f()
        passed = false
        println("      No exception thrown, in exception: $(nb)")
    catch e
        if (isa(e, ExceptionType))
            passed = true
        else
            passed = false
            println("      Wrong exception, in exception: $(nb)")
        end
    end
    return passed
end




end # TestUtils