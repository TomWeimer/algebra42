function Base.getindex(array::NDArray{DType, N}, indices::CopyIndices ) where {DType, N}
    shape = size(array)
    convertedIndices = convertIndices(shape, (indices, ));
    return fancyIndexing(array, convertedIndices)
end

# Fancy indexing (with only array of bool or int)
function fancyIndexing(original_array::NDArray{DType, N}, indices::Tuple{ Vararg{ FancyIndices }} ) where {DType, N}
    # Behavior in numpy: the shape of the indices must be the same
    # Behavior in julia: the shape of the indices must be compatible
    original_shape = size(original_array)


   # println("indices: ", indices, " enter: ", all(index ->  length(size(index)) == 1, indices))
  
    final_shape = all(index ->  length(size(index)) == 1, indices) ?  ntuple(i -> length(indices[i]), length(indices)) : advanced_index_shape(original_shape, indices)

    # println("final_shape: ", final_shape)
        
    check_broadcast_compatible(indices) || throw(DIMENSIONS_DO_NOT_MATCH)

    expandedIndices = expandDimensions( original_shape, indices )

    output_array = NDArray{DType}(final_shape)

    source_grid = CartesianIndices(original_array)[expandedIndices...]
    output_grid = CartesianIndices(output_array)

    for (output_idx, source_idx) in zip(output_grid, source_grid)
        output_array[output_idx] = original_array[source_idx]
    end
    return output_array
end
