function fancy_index(src_array::NDArray, indices)
    # Obtain shape
    src_shape = size(src_array)

    # Obtain only the fancy indices ( [1, 3], [ 1 4; 2 3], ... )
    fancy_indices = _collect_fancy(indices...)

    # We stetch all the shapes of the fancy indices to match a single dimension
    padded_fancy_shapes = obtain_padded_shapes(fancy_indices)

    # Obtain the shape of the output array
    fancy_offset, output_shape = obtain_output_shape_fancy(src_shape, fancy_indices, indices, padded_fancy_shapes)

    # Create the output array
    out = NDArray{eltype(src_array)}(output_shape)

    # Fill the output array
    _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes, fancy_offset)

    return out
end


function _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes::AbstractArray{<:PaddedShape}, fancy_offset)

    # We iterate over the output indices, because the reverse is not one to one 
    output_indices = CartesianIndices_42(output_shape)

    for output_idx in output_indices
        # We compute the source index with the information given by the output_idx
        src_idx = _get_src_idx(indices, output_shape, output_idx, padded_fancy_shapes, fancy_offset)

        println("indices: $(indices...), output_idx: $output_idx, src_idx: $src_idx, src_array: $src_array")
        out[output_idx] = src_array[src_idx...]
    end
end


function _get_src_idx(indices, output_shape, output_idx, padded_shapes::AbstractArray{<:PaddedShape}, fancy_offset)

    # We initialize the full index
    src_idx, output_idx_pos, i = Int[], 1, 1

    for idx in indices

         # We have differents scenario:
         #   1. The index was a scalar in this case a dimension was dropped in the output_idx
         #      so we get it back from the original indices
        if is_fixed(idx)
            push!(src_idx, idx)

        #   2. The index was a fancy index in this we obtain the value of the source index
        #      back using the output_idx 
        elseif is_fancy(idx)
            val = _value_from_fancy(idx, output_shape, output_idx, padded_shapes[i], fancy_offset)
            push!(src_idx, val)
            output_idx_pos += 1
            i+=1
        #   3. The index was a range or a colon in this case the src_index and output_idx have the same value for this axis
        elseif idx isa AbstractRange
            println("idx: ", idx)
           push!(src_idx, output_idx[output_idx_pos] + first(idx) - 1)
           output_idx_pos += 1
        else
           push!(src_idx, output_idx[output_idx_pos])
           output_idx_pos += 1
        end
    end
    return src_idx
end

function _value_from_fancy(idx, output_shape, output_idx, padded_shape::PaddedShape, fancy_offset)

    # We want to obain the value at idx[i, j, k, ...], however it is possible that the
    # index used to access idx is smaller than the ones to access output_idx

    # The first case is trivial, if they have the same shape, they have the same indices
    shape_idx = size(idx)
    if (shape_idx == output_shape)
        return idx[output_idx]
    
    # The second case depends on the broadcast shape, if the padded_shape is not 1 then add the iterator to idx_used
    else
        idx_used = Int[]

        for i in 1:length(padded_shape)
            (padded_shape[i] > 1) && push!(idx_used, output_idx[fancy_offset + i])
        end

        while (length(idx_used) < length(shape_idx))
            push!(idx_used, 1)
        end

       # println("idx: $idx, idx_used: $idx_used, padded_shape: $padded_shape, output_idx: $idx, output_shape: $output_shape")

        return idx[idx_used...]
    end
end


function obtain_output_shape_fancy(src_shape::Tuple, fancy_indices::Tuple, indices::Tuple, padded_shapes::AbstractArray{<:PaddedShape})

    # The shape of the output array depends on two scenario:

    # Scenario 1: All Advanced Indices are Together
    #  If all fancy (advanced) indices are contiguous (no slices, colons, or basic integers separating them), 
    #  broadcasted shape created from the fancy indices replace their place in the array of indices.
    #   - example: A[:, Fancy1, Fancy2] or A[Fancy1, Fancy2, :]  or A[:, Fancy1, Fancy2, :]

    # Scenario 2: Advanced Indices are separated
    #  Otherwise if the fancy indices are separated, they are first broadcasted together this shaoe is S_fancy
    #  then the shape of the other regular index create S_regular
    #  In this second scenerio the output shape is then S_out = (S_fancy, S_regular)
    #   - example: A[:, Fancy1, :, Fancy2] or A[Fancy1, Fancy2, :, 1]  or A[Fancy1, Fancy2, :, Fancy3]
    
    # Obtains the shape broadcasted by the fancy indices
    S_fancy = obtain_broadcast_shape(padded_shapes)
    fancy_offset = 0

    # We are in scenario 1: The fancy indices are together 
    if fancy_indices_are_together(indices)
        output_shape = obtain_output_shape(S_fancy, src_shape, indices)
        fancy_offset = obtain_offset_first_fancy_index(indices)
    # We are in scenario 2: The fancy indices are separated
    else
        # Obtain the shape of all the regular indices
        S_regular = __collect_basic_dims(src_shape, indices, 1)
        output_shape = (S_fancy..., S_regular...)
    end

    return fancy_offset, output_shape
end

function obtain_offset_first_fancy_index(indices)
    return findfirst(is_fancy, indices) - 1
end

function obtain_broadcast_shape(padded_shapes::AbstractArray{<:PaddedShape})
    maxDim = length(padded_shapes[1])

    # We obtain the broadcast shape by taking the maximum of each dimension
    broadcast_shape = ntuple(i -> maximum(ps[i] for ps in padded_shapes), maxDim)

    println("broadcast_shape: ", broadcast_shape)

    # We check that the broadcasted shape is valid
    broadcast_compatible(broadcast_shape, padded_shapes, maxDim) || throw(DomainError("The indices entered are not compatible for broadcasting"))

    return broadcast_shape
end

function broadcast_compatible(broadcast_shape::Tuple, padded_shapes::AbstractArray{<:PaddedShape}, maxDim::Int)
    # The broadcast shape is valid if for every dimension, each input shape's size is either 1 (a singleton dimension, allowing broadcast/repetition)
    # or it exactly matches the final output size for that dimension.
    return all(i -> all(ps[i] == 1 || ps[i] == broadcast_shape[i] for ps in padded_shapes), 1:maxDim)
end
