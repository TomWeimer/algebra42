using Algebra42


function onesArrayTest()
    defaultTest =  (OnesArray((4,)) .* 100) == [100, 100, 100, 100]
    floatTest =    (OnesArray{Float64}((4,)) .* 100) == [100.0, 100.0, 100.0, 100.0]
    return defaultTest && floatTest;
end

function onesVectorTest()
    defaultTest =  (OnesVector((4,)) .* 100) == [100, 100, 100, 100]
    floatTest =    (OnesVector{Float64}((4,)) .* 100) == [100.0, 100.0, 100.0, 100.0]
    return defaultTest && floatTest;
end

function zerosArrayTest()
    defaultTest =  (ZerosArray((4,)) .* 100) == [0, 0, 0, 0]
    floatTest =    (ZerosArray{Float64}((4,)) .* 100) == [0.0, 0.0, 0.0, 0.0]
     return defaultTest && floatTest;
end

function zerosVectorTest()
    defaultTest =  (ZerosVector((4,)) .* 100) == [0, 0, 0, 0]
    floatTest =    (ZerosVector{Float64}((4,)) .* 100) == [0.0, 0.0, 0.0, 0.0]
    return defaultTest && floatTest;
end

function singleOneVector()
     # Test constructor with default type
    sv1 = SingleOneVector(5, 10)

    sv1Passed = size(sv1) == (10,) && sv1[5] == 1 && sv1[4] == 0 && sv1[6] == 0;
    
    # Test constructor with explicit type
    sv2 = SingleOneVector{Float64}(3, 8)

    sv2Passed =  size(sv2) == (8,) && sv2[3] == 1.0 && sv2[2] == 0.0 && sv2[4] == 0.0;
    
    # Test vector operations
    sv3 = SingleOneVector(2, 5)
    result = sv3 * 10
    opPassed = result[2] == 10 && result[1] == 0
    
    return sv1Passed && sv2Passed && opPassed;
end


function run_tests_my_others(write_log_fn)
    @assert onesArrayTest() && onesVectorTest() && zerosArrayTest() && zerosVectorTest() && singleOneVector()
end