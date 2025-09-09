function verifyShape(ndarray, expectedContent::Number)
    shapeMatch =     ( size(ndarray) == (0,) )
    dimensionMatch = ( ndims(ndarray) == 0 )

    shapeMisMatch  || println("Shape mismatch: got $(size(ndarray)), expected $(Base.size(expectedContent))")
    dimensionMatch || println("Dim mismatch: got   $(ndims(ndarray)), expected 0")

    return shapeMatch && dimensionMatch
end