# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                          Indexing Utils                                          #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #


# ═════════════════════════════════ conversion linear/cartesian ══════════════════════════════════ #

function from_linear_to_cartesian(sizeA::NTuple{N, Int}, i::Int; order=ColOrder) where N
    # if empty or scalar then return
    N == 0 && return ()
    inds = Array{Int, 1}(undef, N)
    L_R = i - 1

    iteration_range = order == ColOrder ? (1:N) : (N:-1:1)

    for k in iteration_range
        i_k = L_R % sizeA[k]
        L_R = div(L_R, sizeA[k])
        inds[k] = i_k + 1
    end
    return tuple(inds...)
end

# use strides from the array
Base.@propagate_inbounds function from_cartesian_to_linear(
    a::AbstractArray, I::Union{AllCartesianIndex{N}, NTuple{N, Int}}) where N
    N == 0 && return 1

    strds = strides(a)
    offset = 0
    @inbounds for i in 1:N
        offset += (I[i] - 1) * strds[i]
    end
    return offset + 1
end


# compute column strides on the go
Base.@propagate_inbounds function from_cartesian_to_linear(
    dims::Dims, I::Union{AllCartesianIndex{N}, NTuple{N, Int}}) where N
    N == 0 && return 1
    
    offset = 0
    stride = 1
    @inbounds for i in 1:N
        offset += (I[i] - 1) * stride
        stride *= dims[i]
    end
    return offset + 1
end

# ═══════════════════════════════════════════ strides ════════════════════════════════════════════ #

# Compute Column Major strides

# compute_strides(shape::Shape) = compute_strides(shape.dims)

function compute_strides(dims::Tuple)
    n = length(dims)
    strds = Array{Int, 1}(undef, max(n, 1))
    strds[1] = 1
    @inbounds for i in 2:n
        strds[i] = strds[i-1] * dims[i-1]
    end
    return Tuple(strds)
end

# Merge previous strides with the remaining ones

function compute_strides(previous_strides, dims::Tuple)
    n_prev = length(previous_strides)
    n_dims = length(dims)
    n_missing = n_dims - n_prev
    missing_strides = Array{Int, 1}(undef, n_missing)

    missing_strides[1] = isempty(previous_strides) ? 1 : last(previous_strides) * dims[n_prev]
    for i in 2:n_missing
        missing_strides[i] = missing_strides[i-1] * dims[n_prev + i - 1]
    end

    return (previous_strides..., tuple(missing_strides...)...)
end

# ══════════════════════════════════ NDSubArray strides/offset1 ══════════════════════════════════ #


# ──── substrides ──────────────────────────────────────────────────────────────────────────────── #

# Computes the effective strides of a ndsubarray given the parent array strides
# and the indexing tuple used to create the subarray.

function substrides(parent_strides::NTuple{N, Int}, indices::NTuple{M, Any}) where {N, M}

    strides = N >= M ? parent_strides : ntuple(i -> i <= N ? parent_strides[i] : 1 , M)

    new_strides = []

    for (d, idx) in enumerate(indices)

        # parent index is a real so we skip this stride
        if idx isa Real
            continue
        # Return the stride of the parent nothing chanhed
        elseif idx isa Base.Slice
            push!(new_strides, parent_strides[d])
        elseif idx isa AbstractRange
            push!(new_strides, strides[d] * step(idx))
        else
            throw(ArgumentError("invalid index type $(typeof(idx))"))
        end
    end

    return tuple(new_strides...)
end

# ──── stride1 ─────────────────────────────────────────────────────────────────────────────────── #

# It is the distance between consecutive elements along the fastest-moving dimension 
# (rows in julia/col major )
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
        # - index is a range  (the previous distance is multiplied by the step)
            return s * step(idx)
        elseif idx isa Base.Slice
        # - index is ':' 
        # ( all the elements in the dimension are selected hence the previous distance does not 
        # change further afterwards )
            return s
        else
            throw(ArgumentError("invalid index type $(typeof(idx))"))
        end
    end
    return s
end

# ──── offset1 ─────────────────────────────────────────────────────────────────────────────────── #
            
# It is the linear shift required to map the first element of the view to the corresponding element 
# in the parent array.
# In general we obtain offset1 = position of the first element - stride1

# 1) A simple case is where the parent is a vector:
compute_offset1(::AbstractVector, stride1::Integer, I::Tuple{AbstractRange}) = (
    first(I[1]) - stride1 * first(axes(I[1], 1))
)

# 2)  The parent is a n dimensional array, then this becomes a bit more complex
compute_offset1(parent::AbstractArray, stride1::Integer, I) = (
    _compute_offset1(parent, stride1, extended_dims(I), extended_inds(I), I)
)

# 2A) The only non-scalar indices was a single range
# If the result is one-dimensional and it's a Colon, then linear indexing uses the indices along 
# the given dimension.
_compute_offset1(
    parent::AbstractArray, stride1::Integer, dims::Tuple{Int}, inds::Tuple{Base.Slice}, I) = (
    parent_lindex(parent, I) - stride1 * first(axes(parent, dims[1]))
)

# 2B) The only non-scalar indices was a single slice
# If the result is one-dimensional and it's a range, then linear indexing might be offset if 
# the index itself is offset
_compute_offset1(
    parent::AbstractArray, stride1::Integer, dims::Tuple{Int}, inds::Tuple{AbstractRange}, I) = (
    parent_lindex(parent, I) - stride1 * first(axes(inds[1], 1)) 
)

# 2C) There are multiple non-scalar indices then we use offset1 = view_index - stride1
# Otherwise linear indexing always matches the parent.
_compute_offset1(parent, stride1::Integer, dims, inds, I::Tuple) = (
    parent_lindex(parent, I) - stride1
)

# parent_lindex:
# Computes the linear index of an element in a parent array 
function parent_lindex(parent::AbstractArray, inds::NTuple{N,Any}) where N
    lin = firstindex(parent)
    stride = 1

    axes_parent = Base.fill_to_length(axes(parent), 1:1, Val(N))

    @inbounds for i in 1:N
        lin += (first(inds[i]) - first(axes_parent[i])) * stride
        stride *= length(axes_parent[i])
    end

    return lin
end

# extended_inds
# returns the indices that are non-scalars.
function extended_inds(I::Tuple)
    return tuple(idx for idx in I if !(idx isa Integer))  # skip scalars
end

# extended_dims
# returns the positions (dimension numbers) of those indices
function extended_dims(I::Tuple)
    return tuple(i for (i, idx) in enumerate(I) if !(idx isa Integer))  # skip scalars
end

# ═══════════════════════════════ Reshaped42 indices and dimension ═══════════════════════════════ #

# ──── drop singleton dimension ────────────────────────────────────────────────────────────────── #

# Remove the trailing scalar indices after dimA
function drop_singleton_dimension(indices::NTuple{N, Any}, dimA::Int) where N
    DeadDim = 0

    K = dimA + 1

    # Iterate backwards using the correct indices range
    for i in N:-1:K
        isDeadDim = indices[i] isa Real
        
        if isDeadDim
            DeadDim += 1
        else
            break
        end
    end
  
    # The tuple slicing must go from 1 to N - DeadDim
    return tuple((indices[i] for i in 1:(N - DeadDim))...)
end

# ──── ensure indexable ────────────────────────────────────────────────────────────────────────── #

# Convert boolean array/Array{Bool, 1} to indexable types 

# If there are no indices left, return an empty tuple
ensure_indexable(::Tuple{}) = ()

 #If the first index is a "normal" type (integer, range, or colon :), just leave it as-is.
ensure_indexable(I::Tuple{Any, Vararg{Any}}) = (I[1], ensure_indexable(Base.tail(I))...)

# If the first index is a a boolean array, convert it into a concrete array of indices using collect
ensure_indexable(I::Tuple{Base.LogicalIndex, Vararg{Any}}) = (
    (collect(I[1]), ensure_indexable(Base.tail(I))...)
)

# ──── index dimsum ────────────────────────────────────────────────────────────────────────────── #

# Determines how many non-scalar indices they are and return a tuple of boolean to be inferable

index_dimsum(I...) = ntuple(Returns(true), count(i -> !(isa(i, Real)), I))


# ──── index ndims ─────────────────────────────────────────────────────────────────────────────── #

# Determines which dimensions survive after creating a view from indices

index_ndims() = ()

# If the index is not a scalar or special case, it contributes one dimension
@inline index_ndims(i1, I...) = (true, index_ndims(I...)...)

# Special case: If the index is a Cartesian Index, it contributes one dimension per coordinate
@inline function index_ndims(i1::AllCartesianIndex, I...)
    (map(Returns(true), i1.I)..., index_ndims(I...)...)
end

# Special case: If the index is an n dimensional array of Cartesian Index, it contributes to n dims
@inline function index_ndims(i1::AbstractArray{AllCartesianIndex{N}}, I...) where N
    (ntuple(Returns(true), Val(N))..., index_ndims(I...)...)
end


# ════════════════════════════════════════ fancy indexing ════════════════════════════════════════ #

# ──── is fancy ────────────────────────────────────────────────────────────────────────────────── #

is_fancy(::AbstractArray) = true

is_fancy(::Any) = false
is_fancy(::AbstractRange) = false

# ──── collect fancy ───────────────────────────────────────────────────────────────────────────── #

# Returns only fancy indexing:
_collect_fancy() = ()

_collect_fancy(idx, rest...) =
    is_fancy(idx) ? (idx, _collect_fancy(rest...)...) : _collect_fancy(rest...)


# ──── determines fancy separation ─────────────────────────────────────────────────────────────── #

# Example: A[[1, 3, 5], :, 1:3, []] ( separated case ), A[[1, 3, 5], [2,4], 1:3, :] (But not here)

function fancy_indices_are_together(indices::Tuple)
    seen_fancy = false
    saw_basic_after_fancy = false

    for idx in indices
        # in this case integer are considered fancy index
        if is_fancy(idx) || isa(idx, Real)
            if saw_basic_after_fancy
                return false   # fancy after basic after fancy → separated
            end
            seen_fancy = true
        else
            if seen_fancy
                saw_basic_after_fancy = true
            end
        end
    end
    return true
end

# ──── shape when fancy index are together ─────────────────────────────────────────────────────── #

function fancyshape_together(S_fancy::Tuple, shape_A::Tuple, indices::Tuple)
    return _fshape(S_fancy, shape_A, 1, Val(false), indices...)
end

_fshape(::Tuple, ::Tuple, i::Int, ::Val) = ()

# Integer ( never contributes to dimensions )
_fshape(sf, d, i, seen, idx::Int, rest...) = _fshape(sf, d,  i + 1, seen, rest...)

# Colon index
_fshape(sf, d, i, seen, ::Colon, rest...) = ( d[i], _fshape(sf, d, i + 1, seen, rest...)...)

# Range index
_fshape(sf, d, i, seen::Val{false}, idx::AbstractUnitRange, rest...) = (
    (length(idx), _fshape(sf, d, i + 1, seen, rest...)...)
)

_fshape(sf, d, i, seen::Val{true}, idx::AbstractUnitRange, rest...)  = (
    (length(idx), _fshape(sf, d, i + 1, seen, rest...)...)
)

# fancy index: (Add the broadcasted shape S_fancy when encountering the first fancy index)
_fshape(sf, d,  i, seen::Val{false}, idx::AbstractArray, rest...) = (
    (sf..., _fshape(sf, d, i + 1, Val(true), rest...)...)
)

_fshape(sf, d,  i, seen::Val{true},  idx::AbstractArray, rest...) = (
    _fshape(sf, d,  i + 1, seen, rest...)
)

# ──── Return the shape when fancy index are separated ─────────────────────────────────────────── #

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                           LinearIndex                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# use strides from the array
@propagate_inbounds function LinearIndex(a::AbstractArray, I::Indices{N}) where N
    N == 0 && return 1

    strds = strides(a)
    offset = 0
    @inbounds for i in 1:N
        offset += (I[i] - 1) * strds[i]
    end
    return offset + 1
end


# compute column strides on the go
@propagate_inbounds function LinearIndex(dims::Dims, I::Indices{N}) where N
    N == 0 && return 1
    
    offset = 0
    stride = 1
    @inbounds for i in 1:N
        offset += (I[i] - 1) * stride
        stride *= dims[i]
    end
    return offset + 1
end