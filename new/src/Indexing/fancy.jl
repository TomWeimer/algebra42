# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                         fancy indexing:                                          #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

function fancy_index(src_array::AbstractArray, indices)
    # Obtain shape
    src_shape = size(src_array)

    # Obtain only the fancy indices ( [1, 3], [ 1 4; 2 3], ... )
    fancy_indices = _collect_fancy(indices...)

    # We stetch all the shapes of the fancy indices to match a single dimension
    padded_fancy_shapes = padded_shapes(fancy_indices)

    # Obtain the shape of the output array
    fancy_offset, output_shape = obtain_output_shape_fancy(
        src_shape, 
        fancy_indices, 
        indices, 
        padded_fancy_shapes
    )

    # Create the output array
    out = NDArray{eltype(src_array)}(output_shape)

    # Fill the output array
    _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes, fancy_offset)

    return out
end

# ═════════════════════════════════════════ output shape ═════════════════════════════════════════ #

# According to numpy's rules the resulting shape depends on if the indices are separated or not:
# 1) The fancy indices are together:  
#       the broadcasted shape is placed where the fancy indices were
# 2) The fancy indices are separated:
#       the broadcasted shape is placed at the front of the output shape

function obtain_output_shape_fancy(
    src_shape::Tuple, 
    fancy_indices::Tuple, 
    indices::Tuple, 
    padded_shapes::NTuple{N, PaddedShape}
) where N
    S_fancy = obtain_broadcast_shape(padded_shapes)

    if fancy_indices_are_together(indices)
        output_shape = fancyshape_together(S_fancy, src_shape, indices)
        fancy_offset = findfirst(is_fancy, indices) - 1
    else
        output_shape = fancyshape_separated(S_fancy, src_shape, indices)
        fancy_offset = 0
    end

    return fancy_offset, output_shape
end

# ════════════════════════════════════ fill the output array ═════════════════════════════════════ #

function _fill_fancy!(
    out, 
    src_array, 
    indices, 
    output_shape, 
    padded_fancy_shapes::NTuple{N, PaddedShape}, 
    fancy_offset ) where N

    # We iterate over the output indices, because the reverse is not one to one 
    output_indices = CartesianIndices42(output_shape)

    for output_idx in output_indices
        # We compute the source index with the information given by the output_idx
        src_idx = _get_src_idx(indices, output_shape, output_idx, padded_fancy_shapes, fancy_offset)
        out[output_idx] = src_array[src_idx...]
    end
end

# ──── obtain source index ─────────────────────────────────────────────────────────────────────── #
            
function _get_src_idx(indices, output_shape, output_idx, padded_shapes::NTuple{N, PaddedShape}, 
                      fancy_offset) where N
    
    src_idx  = Array{Int, 1}(undef, length(indices))

    output_idx_pos, fancy_idx_pos = 1, 1

    for (i, idx) in enumerate(indices)
        if isa(idx, Real)
            src_idx[i] = idx
            continue
        elseif is_fancy(idx)
            src_idx[i] = _value_from_fancy(
                idx, 
                output_shape, 
                output_idx, 
                padded_shapes[fancy_idx_pos], 
                fancy_offset
            )
            fancy_idx_pos  += 1
        elseif idx isa AbstractRange
            src_idx[i] = output_idx[output_idx_pos] + first(idx) - 1
        else
            src_idx[i] = output_idx[output_idx_pos]
        end
        output_idx_pos += 1
    end

    return src_idx
end

# ──── obtain fancy index value ────────────────────────────────────────────────────────────────── #

function _value_from_fancy(idx, output_shape, output_idx, padded_shape::PaddedShape, fancy_offset)
    shape_idx = size(idx)
    if (shape_idx == output_shape)
        return idx[output_idx]
    
    else
        idx_used = Int[]

        for i in 1:length(padded_shape)
            (padded_shape[i] > 1) && push!(idx_used, output_idx[fancy_offset + i])
        end

        while (length(idx_used) < length(shape_idx))
            push!(idx_used, 1)
        end

        return idx[idx_used...]
    end
end
