#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                      Broadcasting:                                            #
#_______________________________________________________________________________________________#

# This broadcasting is fairly simple, mainly used in fancy indexing and simple functions like add, sub ...
# For more complex case the julia's implementation is overloaded as it permits the use of the A = B .* C syntax

function broadcast(f::Function, arrays::Vararg{AbstractArray})
    println(stderr, "is called motherfucker")
    shape = obtain_broadcast_shape(arrays)

    output_array = NDArray{eltype(first(arrays))}(shape)

    # Iterate over broadcasted arrays
    for idx in CartesianIndices(shape)
        vals = ntuple(i -> arrays[i][ broadcast_index(idx, size(arrays[i]))... ], length(arrays) )
        output_array[idx] = f(vals...)
    end
    return output_array
end

# ======== Broadcast shape ==================================================================== #

obtain_broadcast_shape(arrays::Tuple{Vararg{Number}}) = (
    return ntuple(i -> i, 0)
)

obtain_broadcast_shape(arrays::Tuple) = (
    shapes = padded_shapes(arrays);
    obtain_broadcast_shape(shapes)
)

function obtain_broadcast_shape(padded_shapes::NTuple{N, PaddedShape{<:Any, MaxDim}}) where {N, MaxDim}

    # We obtain the broadcast shape by taking the maximum of each dimension
    broadcast_shape = ntuple(i -> maximum(ps[i] for ps in padded_shapes), MaxDim)

    # We check that the broadcasted shape is valid
    all(i -> all(ps[i] == 1 || ps[i] == broadcast_shape[i] for ps in padded_shapes), 1:MaxDim) || throw( 
        DomainError("The indices entered are not compatible for broadcasting")
    )

    return broadcast_shape
end

# replace indices of stretched dimensions by 1
broadcast_index(idx::CartesianIndex, sizeA::Tuple) = (
    ntuple(d -> sizeA[d] == 1 ? 1 : idx[d], length(sizeA))
)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                      Fancy indexing:                                          #
#_______________________________________________________________________________________________#

# A fancy index is any AbstractArray (Vector, BitArray, etc.)

# ======== Process the fancy index case ==================================================================== #

function fancy_index(src_array::AbstractArray, indices)
    # Obtain shape
    src_shape = size(src_array)

    # Obtain only the fancy indices ( [1, 3], [ 1 4; 2 3], ... )
    fancy_indices = _collect_fancy(indices...)

    # We stetch all the shapes of the fancy indices to match a single dimension
    padded_fancy_shapes = padded_shapes(fancy_indices)

    # Obtain the shape of the output array
    fancy_offset, output_shape = obtain_output_shape_fancy(src_shape, fancy_indices, indices, padded_fancy_shapes)

    # Create the output array
    out = NDArray{eltype(src_array)}(output_shape)

    # Fill the output array
    _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes, fancy_offset)

    return out
end

# ======== Obtain the shape of the result  ==================================================================== #

# According to numpy's rules when the fancy indices are separated or not the resulting shape differs:
# If the fancy indices are together:  the broadcasted shape of those indices is placed where they were
# If the fancy indices are separated: the broadcasted  "    "   "      "     is placed in front
function obtain_output_shape_fancy(src_shape::Tuple, fancy_indices::Tuple, indices::Tuple, padded_shapes::NTuple{N, PaddedShape}) where N
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

# ======== Fill the output array ==================================================================== #

function _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes::NTuple{N, PaddedShape}, fancy_offset) where N
    # We iterate over the output indices, because the reverse is not one to one 
    output_indices = CartesianIndices_42(output_shape)

    for output_idx in output_indices
        # We compute the source index with the information given by the output_idx
        src_idx = _get_src_idx(indices, output_shape, output_idx, padded_fancy_shapes, fancy_offset)
        out[output_idx] = src_array[src_idx...]
    end
end

# ======== Recreates the full source index ==================================================================== #

function _get_src_idx(indices, output_shape, output_idx, padded_shapes::NTuple{N, PaddedShape}, fancy_offset) where N

    src_idx  = Array{Int, 1}(undef, length(indices))

    output_idx_pos, fancy_idx_pos = 1, 1

    for (i, idx) in enumerate(indices)
        if isa(idx, Real)
            src_idx[i] = idx
            continue
        elseif is_fancy(idx)
            src_idx[i] = _value_from_fancy(idx, output_shape, output_idx, padded_shapes[fancy_idx_pos], fancy_offset)
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

# ======== Obtain the values from the fancy index ==================================================================== #

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
