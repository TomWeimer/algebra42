function obtain_output_shape_fancy(src_shape::Tuple, fancy_indices::Tuple, indices::Tuple, padded_shapes::AbstractArray{<:PaddedShape})
    
    S_fancy = obtain_broadcast_shape(padded_shapes)

    # We are in scenario 1: The fancy indices are together 
    if fancy_indices_are_together(indices)
        output_shape = fancyshape_together(S_fancy, src_shape, indices)
        fancy_offset = findfirst(is_fancy, indices) - 1
    # We are in scenario 2: The fancy indices are separated
    else
        output_shape = fancyshape_separated(S_fancy, src_shape, indices)
        fancy_offset = 0
    end

    return fancy_offset, output_shape
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                             Obtain broadcast shape from fancy indices:                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function obtain_broadcast_shape(padded_shapes::AbstractArray{<:PaddedShape})
    maxDim = length(padded_shapes[1])

    # We obtain the broadcast shape by taking the maximum of each dimension
    broadcast_shape = ntuple(i -> maximum(ps[i] for ps in padded_shapes), maxDim)

    # We check that the broadcasted shape is valid
    all(i -> all(ps[i] == 1 || ps[i] == broadcast_shape[i] for ps in padded_shapes), 1:maxDim) || throw( 
        DomainError("The indices entered are not compatible for broadcasting")
    )

    return broadcast_shape
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                             Obtain dimensions for fancy indexing case A                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Obtain the output shape from the indices (used in fancy indexing)
function fancyshape_together(S_fancy::Tuple, shape_A::Tuple, indices::Tuple)
    # the output shape in this scenario is (dim1, ..., S_fancy, ..., dimN) 
    return _fancyshape(S_fancy, shape_A, 1, Val(false), indices...)
end

# Base case: no more indices left
_fancyshape(::Tuple, ::Tuple, i::Int, ::Val) = ()

# fancy index not found yet: Add dimension of colon index
_fancyshape(sf::Tuple, d::Tuple, i::Int, ::Val{false}, idx::Colon, rest...) = 
( d[i], _fancyshape(sf, d, i + 1, Val(false), rest...)...)

# fancy index not found yet: Add dimension of range index
_fancyshape(sf::Tuple, d::Tuple, i::Int, ::Val{false}, idx::AbstractUnitRange, rest...) =
    (length(idx), _fancyshape(sf, d, i + 1, Val(false), rest...)...)

# fancy index not found yet: First fancy index encountered add the precomputed shape made by all fancy indices
_fancyshape(sf::Tuple, d::Tuple,  i::Int, ::Val{false}, idx::AbstractArray, rest...) =
    (sf..., _fancyshape(sf, d, i + 1, Val(true), rest...)...)

# fancy index already found: Another fancy index, skip it 
_fancyshape(sf::Tuple, d::Tuple,  i::Int, ::Val{true}, idx::AbstractArray, rest...) =
    _fancyshape(sf, d,  i + 1, Val(true), rest...)

# fancy index already found: Add dimension of colon index
_fancyshape(sf::Tuple, d::Tuple, i::Int, ::Val{true},  idx::Colon, rest...) =
    (d[i], _fancyshape(sf, d, i + 1, Val(true), rest...)...)

# fancy index already found: Add dimension of range index
_fancyshape(sf::Tuple, d::Tuple, i::Int, ::Val{true}, idx::AbstractUnitRange, rest...) =
    (length(idx), _fancyshape(sf, d,  i + 1, Val(true), rest...)...)

# Skip integer index do not contribute to shape
_fancyshape(sf::Tuple, d::Tuple, i::Int, flag::Val, idx::Int, rest...) =
    _fancyshape(sf, d,  i + 1, flag, rest...)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                             Obtain dimensions for fancy indexing case B                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Obtain the output shape from the indices (used in fancy indexing)
function fancyshape_separated(S_fancy::Tuple, src_shape::Tuple, indices::Tuple)
    # the output shape in this scenario is (S_fancy, ..., S_regular) 
    S_regular = _regularshape(S_fancy, src_shape, 1, Val(false), indices...)
    output_shape = (S_fancy..., S_regular...)
    return output_shape
end

# Base Case: There is no remaining indices, return an empty tuple
_regularshape(::Tuple, ::Int) = ()

# Add dimension of range
_regularshape(d::Tuple, i::Int, idx::AbstractUnitRange, rest...) = 
    (length(idx), _regularshape(d, i+1, rest...)...)

# Add dimension of colon
_regularshape(d::Tuple, i::Int, idx::Colon, rest...) = 
    (d[i], _regularshape(d, i+1, rest...)...)

# Scalar and fancy index do not particpate to the regular shape
_regularshape(d::Tuple, i::Int, idx::Union{Int, AbstractArray}, rest...) = 
    _regularshape(d, i + 1, rest...)

