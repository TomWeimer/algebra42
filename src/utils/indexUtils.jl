
# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                         Obtain the src index from dest index:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function _get_src_idx(indices, output_shape, output_idx, padded_shapes::AbstractArray{<:PaddedShape}, fancy_offset)

    src_idx  = Array{Int, 1}(undef, length(indices))

    output_idx_pos, fancy_idx_pos = 1, 1

    for (i, idx) in enumerate(indices)
        if is_fixed(idx)
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

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Obtain only fancy indices:                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Case 2: The first index is a regular index (Colon, Int, Range) -> ignore the first index and continue to filter the rest
_collect_fancy(::AbstractUnitRange, rest...) = _collect_fancy(rest...)

# Case 1: The first index is a fancy index (AbstractArray) -> keep the first index and continue to filter the rest
_collect_fancy(idx::AbstractArray, rest...) = (idx, _collect_fancy(rest...)...)

# Case 2: The first index is a regular index (Colon, Int, Range) -> ignore the first index and continue to filter the rest
_collect_fancy(idx::Union{Int, Colon, AbstractUnitRange}, rest...) = _collect_fancy(rest...)

# Final Case: There is no remaining indices, return an empty tuple
_collect_fancy() = ()

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                          Determine if fancy indices are next to each other:                   #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Determines fancy indices:

# A fancy index is any AbstractArray (Vector, BitArray, etc.)
is_fancy(::AbstractRange) = false

is_fancy(::AbstractArray) = true

is_fancy(::Any) = false # Colon, Int, UnitRange, etc., are considered basic

is_fixed(::Int) = true # only single integer drop dimension
is_fixed(::Any) = false # Colon, AbstractArray, UnitRange, etc., are considered not fixed

# A basic index is NOT a fancy index
is_basic(idx) = !is_fancy(idx)

# Dispatcher:

# This dispatcher returns two states in contrast to the previous ones
# State 1: has_seen_fancy - Has an advanced index been encountered yet?
# State 2: has_seen_basic_after_fancy - Has a basic index been encountered after the first fancy index?


# Case A: Found a Fancy Index (First one, or consecutive) but has not seen yet a regular index
# Set has_seen_fancy=true and continue checking the remaining indices.
_check_for_separation(::Val{false}, ::Val{false}, idx::AbstractArray, rest...) = 
    _check_for_separation(Val{true}(), Val{false}(), rest...)

# Case A: Found a Fancy Index (First one, or consecutive) but has not seen yet a regular index
# Set has_seen_fancy=true and continue checking the remaining indices.
_check_for_separation(::Val{false}, ::Val{false}, idx::AbstractUnitRange, rest...) = 
    _check_for_separation(Val{true}(), Val{false}(), rest...)

# Case B: Found a regular Index but has not seen yet a fancy index
# State doesn't change yet, just recurse.
_check_for_separation(has_fancy, has_basic_after_fancy, idx::Union{Int, Colon, AbstractUnitRange}, rest...) =
    _check_for_separation(has_fancy, has_basic_after_fancy, rest...)

# Case E: 
_check_for_separation(::Val{true}, ::Val{false}, idx::AbstractArray, rest...) =
    _check_for_separation(Val{true}(), Val{false}(), rest...)

_check_for_separation(::Val{true}, ::Val{false}, idx::AbstractUnitRange, rest...) =
    _check_for_separation(Val{true}(), Val{true}(), rest...)

# Case C: Found a Basic Index after a fancy index
# Switch the has_seen_basic_after_fancy flag to true and recurse.
_check_for_separation(::Val{true}, ::Val{false}, idx::Union{Int, Colon, AbstractUnitRange}, rest...) =
    _check_for_separation(Val{true}(), Val{true}(), rest...)

# Case D: SEPARATION DETECTED!
# We've seen a fancy index AND we've seen a basic index after it. If the current index is FANCY, we have separation.
_check_for_separation(::Val{true}, ::Val{true}, idx::AbstractArray, rest...) = true

# Case G: SEPARATION DETECTED!
# We've seen an integer AND we've seen a basic index after it. If the current index is FANCY, we have separation.
_check_for_separation(::Val{true}, ::Val{true}, idx::Int, rest...) = true

# Final Case: End of recursion: No separation found.
_check_for_separation(::Val, ::Val) = false

function fancy_indices_are_together(indices::Tuple)
    # Start the recursive check
    return !_check_for_separation(Val{false}(), Val{false}(), indices...)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Strides:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Compute Column Major strides from shape
function compute_strides(dims::Tuple)
    n = length(dims)
    strds = Array{Int, 1}(undef, max(n, 1))
    strds[1] = 1
    @inbounds for i in 2:n
        strds[i] = strds[i-1] * dims[i-1]
    end
    return Tuple(strds)
end

compute_strides(shape::Shape) = compute_strides(shape.dims)

function compute_strides(parent::AbstractArray, indices, N::Int)
    new_strides_list = Array{Int, 1}(undef, N)

    strds = strides(parent)

    i = 1
    for idx in indices
        # Dimension Dropping: Do nothing, don't include in shape/strides
        if idx isa Integer
            continue           
        else
        # Compute strides and shape from the indices
            step_k = idx isa AbstractRange ?  abs(step(idx)) : 1
            new_stride_k = strds[i] * step_k
            new_strides_list[i] = new_stride_k
            i+=1
        end
        if (i > N)
            break
        end
    end

    # Convert lists to tuples and determine the new rank N_new
    return tuple(new_strides_list...)
end



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Offset:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Use multiple indices to compute their corresponding offset (linear index) 

# Scalar
_offset(::AbstractArray{T, 0}, ::Nothing; default_offset::Int = 0) where T = default_offset + 1

# CartesianIndex
_offset(a::AbstractArray{T, N}, I::Base.AbstractCartesianIndex{N}; default_offset::Int = 0) where {T, N} = (
    _offset(strides(a), Tuple(I); default_offset = default_offset)
)

# 'Normal' indices
_offset(a::AbstractArray{T, N}, indices::NTuple{N, Int}; default_offset::Int = 0) where {T, N} = (
    _offset(strides(a), indices; default_offset = default_offset)
)

# Metadata
function _offset(strides::NTuple{N, Int}, I::Base.AbstractCartesianIndex{N}; default_offset::Int = 0) where {N}
    return _offset(strides, Tuple(I); default_offset = default_offset )
end

# Metadata
function _offset(strides::NTuple{N, Int}, indices::NTuple{N, Int}; default_offset::Int = 0) where {N}
   offset = default_offset
   for i in 1:N
      offset += (indices[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1
end

function _offset(strides, indices::NTuple{N, Int}; default_offset::Int = 0) where {N}
   offset = default_offset
   for i in 1:N
      offset += (indices[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                             from Linear to parent Cartesian:                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Index de base : On commence par $L' = L - 1$ (pour travailler en 0-basé).
# Itération : Pour chaque dimension $d$ de 1 à $N_V$ 
# - Taille : Récupérer la taille $s_d$ de la dimension $d$ (c'est-à-dire $size(V, d)$).
# - Index id​ :$$i_d = 1 + L' \pmod{s_d}$$
# - Mise à jour :$$L' = \lfloor L' / s_d \rfloor$$
# Résultat : Les indices $(i_1, i_2, \dots, i_{N_V})$ sont les indices cartésiens dans la vue $V$.

# works only for view

function fromLinearToCartesian(L::Int, a::AbstractArray{T, N}) where {T, N}
    L_0 = L - 1

    indices = []

    for d in 1:N
        s_d = size(a, d)
        i_d = Base.mod(L_0, s_d) 
        push!(indices, i_d + 1)
        L_0 = div(L_0, s_d) # division entiere
    end
    I =  CartesianIndex(_toParentIndices(indices, a.indices, Base.ndims(parent(a)))...)
    return I
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                from Linear to parent Linear:                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


function fromLinearToParentLinear(L::Int, a::AbstractArray{T, N}) where {T, N}
    
    L_0 = L - 1

    indices = []

    # N_parent = ndims(a.parent)

    for d in 1:N
        s_d = size(a, d)
        i_d = Base.mod(L_0, s_d) 
        push!(indices, i_d + 1)
        L_0 = div(L_0, s_d) # division entiere
    end

    I =  CartesianIndex(indices...)
    parent_i =  _offset(strides(a), I; default_offset = a.offset1)

    #     # 1. Get the Cartesian index corresponding to the linear index L
    # I = CartesianIndices(axes(a))[L]

    # # 2. Get the Cartesian indices of the Parent array corresponding to I
    # parent_I = parentindices(a, I)

    # # 3. Convert the Parent Cartesian index to a Parent Linear index
    # parent_L = LinearIndices(parent(a))[parent_I]
    return parent_i
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                from Cartestian to parent Cartesian:                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# works only for view
function _toParentIndices(cinds, sliced_indices::NTuple{N, Any}, N_parent::Int) where{T, N}
    finalIndices = []

    cpos = 1
    for i in 1:N_parent
        idx = sliced_indices[i]
        
        if idx isa Integer
            push!(finalIndices, idx)
        else
            cindex = cinds[cpos]
            push!(finalIndices, idx[cindex])
            cpos += 1
        end
    end
    return finalIndices
end



# works only for view
function _toParentCartesian(a::AbstractArray{T, N}, I::Base.AbstractCartesianIndex{N}, N_parent::Int) where{T, N}
    finalIndices = []
    cpos = 1

    for i in 1:N_parent
        idx = a.indices[i]
        
        if idx isa Integer
            push!(finalIndices, idx)
        else
            val_idx = I[cpos]
            push!(finalIndices, idx[val_idx])
            cpos += 1
        end
    end
    return CartesianIndex(finalIndices...)
end

function compute_offset(strides::NTuple{N, Int}, inds) where {N}
    N_inds = length(inds)
    extendedInds = N_inds >= N ? inds : ntuple(i -> i <= N_inds ? inds[i] : 1 , N)

    return sum((starting_idx(extendedInds[k]) - 1) * strd for (k, strd) in enumerate(strides))
end

# function compute_offset1(A::AbstractArray, inds::NTuple{N, Any}) where N
#     offset = 0
#     axesA = axes(A)
#     stridesA = strides(A)
#     for k in 1:length(inds)
#         I = inds[k]
#         if isa(I, Integer)
#             offset += (I - first(axesA[k])) * stridesA[k]
#         end
#     end
#     return offset
# end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Computation of Linear indices:                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Position(Local index):
# -----------------------

# In Julia with array (contingous memory), the position of an element is determined by:
#                                   local index = stride1 + offset1

# Linear index:
# ------------
function linear_index(parent, I::NTuple{N,Any}) where N
    parent_inds = Base.fill_to_length(axes(parent), 1:1, Val(N))
    return linear_index(parent, parent_inds, I)
end

function linear_index(parent, parent_inds::Tuple, inds::Tuple{Any, Vararg{Any}})
    stride = 1
    linear_index = firstindex(parent)
    
    for (i, idx) in enumerate(inds)
        parent_idx = parent_inds[i]
        Δi = first(idx) - first(parent_idx)
        linear_index += Δi * stride
        stride *= length(parent_idx)
    end
    return linear_index
end

# Stride1:
# --------

# stride1 is the distance between consecutive elements along fastest-moving dimension (rows) = first dimension.
# This distance is computed form the indices received at the creation of the view:
function compute_stride1(parent::AbstractArray, I::NTuple{N,Any}) where N
    s = 1
    N_parent = ndims(parent)

    axes_tmp = axes(parent)
    axes_parent = N_parent >= N ? axes_tmp : ntuple(i -> i <= N_parent ? axes_tmp[i] : 1 , N)
    
    # This distance is the sum of how much each indices contributes, depending on their type
    for (d, idx) in enumerate(I)
        if idx isa Real
        # - index are scalar -> stride1 += size of the parent dimension
            s *= length(axes_parent[d])
        elseif idx isa AbstractRange
        # - index are range  -> stride1 = stride1 * step(range) (the previous distance is multiplied by the step)
            return s * step(idx)
        elseif idx isa Base.Slice
        # - index are range  -> stride1 = stride1 ( all the elements in the dimension are selected which means that the previous distance does not change further with the next indices )
            return s
        else
            throw(ArgumentError("invalid index type $(typeof(idx))"))
        end
    end

    return s
end

# Offset1:
# --------
# The linear shift required to map the first element of the view to the corresponding element in the parent array.

# In general we can obtain offset1 = position of the first element - stride1

# 1) A simple case is where the parent is a vector:
compute_offset1(::AbstractVector, stride1::Integer, I::Tuple{AbstractRange}) = (
    first(I[1]) - stride1 * first(axes(I[1], 1))
)

# 2)  The parent is a n dimensional array, then this becomes a bit more complex
compute_offset1(parent::AbstractArray, stride1::Integer, I) = (
    _compute_offset1(parent, stride1, extended_dims(I), extended_inds(I), I)
)

# 2A) The only non-scalar indices was a single range
# If the result is one-dimensional and it's a Colon, then linear indexing uses the indices along the given dimension.
_compute_offset1(parent::AbstractArray, stride1::Integer, dims::Tuple{Int}, inds::Tuple{Base.Slice}, I) = (
    linear_index(parent, I) - stride1 * first(axes(parent, dims[1]))
)

# 2B) The only non-scalar indices was a single slice
# If the result is one-dimensional and it's a range, then linear indexing might be offset if the index itself is offset
_compute_offset1(parent::AbstractArray, stride1::Integer, dims::Tuple{Int}, inds::Tuple{AbstractRange}, I) = (
    linear_index(parent, I) - stride1 * first(axes(inds[1], 1)) 
)

# 2C) There are multiple non-scalar indices then we use offset1 = view_index - stride1
# Otherwise linear indexing always matches the parent.
_compute_offset1(parent, stride1::Integer, dims, inds, I::Tuple) = (
    linear_index(parent, I) - stride1
)

function extended_dims(I::Tuple)
    return tuple(i for (i, idx) in enumerate(I) if !(idx isa Integer))  # skip scalars
end

# Return a tuple of non-scalar indices themselves
function extended_inds(I::Tuple)
    return tuple(idx for idx in I if !(idx isa Integer))  # skip scalars
end

# Return the starting index for various types
starting_idx(idx::Integer)      = idx
starting_idx(r::AbstractRange)  = first(r)
starting_idx(::Base.Slice)      = 1
starting_idx(::Colon)           = 1
starting_idx(::Any)             = error("Unsupported index type")



function drop_singleton_dimension(indices::NTuple{N, Any}, dimA::Int) where N
    DeadDim = 0

    K = dimA + 1

    # Iterate backwards using the correct indices range
    for i in N:-1:K
        # In actual Julia, this is checked via traits (e.g., is_always_indexed_with_a_scalar)
        # We model this by checking if the type is a plain integer.
        isDeadDim = indices[i] isa Integer
        
        if isDeadDim
            DeadDim += 1
        else
            break
        end
    end
  
    # The tuple slicing must go from 1 to N - DeadDim
    I =  tuple((indices[i] for i in 1:(N - DeadDim))...)

    return I
end

@inline index_ndims(i1, I...) = (true, index_ndims(I...)...)
@inline function index_ndims(i1::CartesianIndex, I...)
    (map(Returns(true), i1.I)..., index_ndims(I...)...)
end
@inline function index_ndims(i1::AbstractArray{CartesianIndex{N}}, I...) where N
    (ntuple(Returns(true), Val(N))..., index_ndims(I...)...)
end
index_ndims() = ()


function index_dimsum(I...) where N
    ndims = 0

    for idx in I
        if !(idx isa Real)
        ndims += 1
        end
    end
    return ntuple(Returns(true), ndims)
end

# If there are no indices left, return an empty tuple
ensure_indexable(::Tuple{}) = ()

# If the first index is a normal thing (like :, 1, or a range),
# keep it as-is and recurse on the rest
ensure_indexable(I::Tuple{Any, Vararg{Any}}) = (I[1], ensure_indexable(tail(I))...)

# If the first index is a logical mask (like BitVector or Vector{Bool}),
# convert it to a regular array of indices (1s and 0s) using collect(),
# then recurse
ensure_indexable(I::Tuple{Base.LogicalIndex, Vararg{Any}}) = (collect(I[1]), ensure_indexable(tail(I))...)


function index_shape(indices::NTuple{N, Any}) where N
    # Filter the indices, keeping only those that are NOT ScalarIndex
    # A ScalarIndex (like an Int) signals a dimension should be dropped.
    shape = []

    for idx in indices
        dim = length_index(idx)
        if dim > 0
            push!(shape, dim)
        end 
    end
    return tuple(shape...)
end


length_index(idx::Int) = 0
length_index(idx::AbstractRange) = length(idx)
length_index(idx::AbstractArray) = length(idx)
length_index(idx::Base.AbstractCartesianIndex{N}) where N = N
length_index(::Any) = throw(ArgumentError("This type is not accepted as an index"))


function parentindices(indices, view_indices, ::AbstractArray{T, P_DIM}) where {T, P_DIM}
    
    full_idx = Array{Any, 1}(undef, P_DIM)

    i = 1
    # position of the last index used
    indices_pos = 1
    
    # We go through all indices used when creating the view
    # If a dimension was dropped (scalar index) we reinsert it back
    # Otherwise we insert the value associated to the right index
    for idx in view_indices
        if (idx isa Real)
            full_idx[i] = idx
        else
            value = indices[indices_pos]
            full_idx[i] = idx[value]
            indices_pos += 1
        end
        i+=1
    end

    # we add the missing indices
    while i <= P_DIM
        full_idx[i] = 1
        i+=1
    end

    return tuple(full_idx...)
end

function parentcartesian(I::CartesianIndex, view_indices, parent::AbstractArray)    
    cartesian = CartesianIndex( parentindices(Tuple(I), view_indices, parent)... )
    return cartesian
end

function parentcartesian(indices::NTuple{N, Any}, view_indices, parent::AbstractArray) where N    
    cartesian = CartesianIndex( parentindices(indices, view_indices, parent)... )
    return cartesian
end

function parentcartesian(sizeA::NTuple{N, Int}, L::Int, view_indices, parent::AbstractArray) where {T, N}    
    cartesian_indices = _linear_to_cartesian(sizeA, L)
    return CartesianIndex( parentindices(cartesian_indices, view_indices, parent)... )
end


function parentindex(sizeA, stridesA::NTuple{N, Int}, L::Int, L_start::Int) where {T, N}
    cartesian_indices = _linear_to_cartesian(sizeA, L)
    # L_parent = L_start + indices[i] * strides_View[i]
    L_parent = _offset(stridesA, cartesian_indices; default_offset = L_start)
    return L_parent
end

function _linear_to_cartesian(sizeA, L::Int)
    L_0 = L - 1

    N = length(sizeA)

    cartesian_indices = Array{Int, 1}(undef, N)

    for d in 1:N
        s_d = sizeA[d]
        i_d = Base.mod(L_0, s_d) 
        cartesian_indices[d] = i_d + 1
        L_0 = div(L_0, s_d) # division entiere
    end
    return tuple(cartesian_indices...)
end

function compute_row_major_strides(sizeA::NTuple{N, Int}) where N
    strds = Array{Int, 1}(undef, max(N, 1))
    
    if N != 0
        strds[N] = 1
        for i in (N-1):-1:1
            strds[i] = strds[i+1] * sizeA[i+1]
        end
    end
    return tuple(strds...)
end

# row-major
function _linear_to_cartesian_row_major(sizeA, i::Int)
    i_0 = i - 1

    strides = compute_row_major_strides(sizeA)

    N = length(sizeA)

    cartesian_indices = Array{Int, 1}(undef, N)

    for j in 1:N
        x = div(i_0, strides[j]) 
        i_j = Base.mod(x, sizeA[j])
        cartesian_indices[j] = i_j + 1
    end
    return tuple(cartesian_indices...)
end


# 1. Get the Cartesian index corresponding to the linear index L
# I = CartesianIndices(axes(a))[L]

# # 2. Get the Cartesian indices of the Parent array corresponding to I
# parent_I = parentindices(a, I)

# # 3. Convert the Parent Cartesian index to a Parent Linear index
# parent_L = LinearIndices(parent(a))[parent_I]
