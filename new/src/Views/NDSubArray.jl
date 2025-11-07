using .Errors: ERR_VIEW_INDICES_PARENT_DIM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                            NDSubArray                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct NDSubArray{T,N,P,I,L} <: AbstractNDArray{T,N}
    parent::P
    indices::I

    offset1::Int                   # for linear indexing and pointer, only valid when L==true
    stride1::Int                   # used only for linear indexing


    function NDSubArray{T,N,P,I,L}(parent, indices, offset1, stride1) where {T,N,P,I,L}
        @inline
        check_view_indices(parent, indices)
        new(parent, indices, offset1, stride1)
    end
end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #
    
const NDScalarView{T,P,I,L} = NDSubArray{T,0,P,I,L}
const NDSubArrayFast{T,N,P,I} = NDSubArray{T,N,P,I,true}
const NDSubArraySlow{T,N,P,I} = NDSubArray{T,N,P,I,false}

# ──── constructors ────────────────────────────────────────────────────────────────────────────── #

NDSubArray(parent::AbstractArray, indices) = NDSubArray(
    IndexStyle(Base.viewindexing(indices), IndexStyle(parent)),
    parent,
    ensure_indexable(indices),
    index_dimsum(indices...)
)

NDSubArray(a::NDSubArray{T,N,P,I,L}) where {T,N,P,I,L} = (
    NDSubArray{T,N,P,I,L}(a.parent, a.indices, a.offset1, a.stride1)
)

NDSubArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N} = (
    stride = compute_stride1(parent, indices);
    offset = compute_offset1(parent, stride, indices);
    NDSubArray{eltype(P),N,P,I,true}(parent, indices, offset, stride)
)

NDSubArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N} = (
    NDSubArray{eltype(P),N,P,I,false}(parent, indices, 0, 0)
)

# ══════════════════════════════════ implement AbstractNDArray ═══════════════════════════════════ #

# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.axes(a::NDSubArray) = _axes(a)
Base.parent(array::NDSubArray) = array.parent

# ──── Internal functions ──────────────────────────────────────────────────────────────────────── #

@propagate_inbounds _getElement(a::NDScalarView{T}) where T = __getElement(a)
@propagate_inbounds _setElement!(a::NDScalarView{T}, val::T) where T = __setElement!(a, val)

# Linear index
@propagate_inbounds _getElement(a::NDSubArray{T,N}, i::Int) where {T,N} = (
    __getElement(a, i)
)

@propagate_inbounds _setElement!(a::NDSubArray{T,N}, val::T, i::Int) where {T,N} = (
    __setElement!(a, val, i)
)

# Cartesian index
@propagate_inbounds _getElement(a::NDSubArray{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    __getElement(a, I)
)

@propagate_inbounds _setElement!(a::NDSubArray{T,N}, val, I::AllCartesianIndex{N}) where {T,N} = (
    __setElement!(a, val, I)
)

# 'Normal' index
@propagate_inbounds _getElement(a::NDSubArray{T,N}, indices::NTuple{N,Index}) where {T,N} = (
    __getElement(a, indices)
)

@propagate_inbounds _setElement!(a::NDSubArray{T,N}, val, indices::NTuple{N,Index}) where {T,N} = (
    __setElement!(a, val, indices)
)

# ═════════════════════════════════ functions for all NDSubArray ═════════════════════════════════ #
            
# ──── core properties ─────────────────────────────────────────────────────────────────────────── #

Base.pointer(array::NDSubArray) = pointer(parent(array))
Base.strides(V::NDSubArray) = substrides(strides(parent(V)), V.indices)

# ──── view creation ───────────────────────────────────────────────────────────────────────────── #

# By default AbstractNDArray use SubNDArray when creating a view
Base.view(a::AbstractNDArray, inds...) = create_view(a, inds...)

# When doing a view of a view, we avoid stacking trivial view
Base.view(a::NDSubArray, inds...) = (
    is_trivial_view(a, inds...) ? NDSubArray(a) : create_view(a, inds...)
)

# When creating a view:
# 1. translate to formatted indices
# 2. drop all trailing scalar indices bigger than ndims(a)
# 3. reshape the parent if needed (change of dimensionality between the view and the parent)
# And finally return a NDSubArrray (the view)
function create_view(a::AbstractNDArray, inds...)
    I = to_indices(a, inds)
    @boundscheck checkbounds(a, I...)
    I_2 = drop_singleton_dimension(I, ndims(a))
    reshaped_parent = maybe_reshape_parent(a, Base.index_ndims(I_2...))
    return _create_view(reshaped_parent, I_2...)
end

# Create a simple view if A was not already a view
_create_view(A::AbstractArray, I::Vararg{Any,N}) where {N} = NDSubArray(A, I)

# Try to reindex the indices and pass the parent directly to avoid stacking view 
_create_view(V::NDSubArray, I::Vararg{Any,N}) where {N} = maybe_reindex(V, I...)

# ═══════════════════════════════════════ reindex indices ════════════════════════════════════════ #
# ──── choose to reindex or not ────────────────────────────────────────────────────────────────── #

maybe_reindex(V::NDSubArray, I...) = (
    all(is_reindexable, I) ? NDSubArray(V.parent, reindex(V.indices, I)) : NDSubArray(V, I)
)

is_reindexable(idx::Any) = true
is_reindexable(idx::AbstractArray{<:AllCartesianIndex{1}}) = true

# Only arrays containing cartesian index with ndims > 1 are not reindexable
is_reindexable(idx::AbstractArray{<:AllCartesianIndex{N}}) where N = false

# ──── reindex ─────────────────────────────────────────────────────────────────────────────────── #

function reindex(parentindices, subindices)
    new_indices = []
    subindices_pos = 1

    for idx in parentindices

        # if parent idx have scalar simply add them
        if idx isa Real
            push!(new_indices, idx)
            # if parent idx is a slice pass children idx
        elseif idx isa Base.Slice
            push!(new_indices, subindices[subindices_pos])
            subindices_pos += 1
            # if parent idx is an abstract array of dim n we expect to get n indices in the 
            # subindices
        else
            N = ndims(idx)
            if (subindices_pos + N - 1 <= length(subindices))
                child_indices = [subindices[i] for i in subindices_pos:(subindices_pos+N-1)]
                push!(new_indices, idx[child_indices...])
                subindices_pos += N
            else
                s = "cannot re-index NDSubArray with fewer indices ($N) than dimensions"
                throw(ArgumentError(s))
            end
        end

    end
    return tuple(new_indices...)
end


# ──── get/set index ───────────────────────────────────────────────────────────────────────────── #

__getElement(a::NDScalarView) = parent(a)[_getIndex(a)...]
__setElement!(a::NDScalarView, val) = parent(a)[_getIndex(a)...] = val

__getElement(a::NDSubArray, idx) = parent(a)[_getIndex(a, idx)...]
__setElement!(a::NDSubArray, val, idx) = parent(a)[_getIndex(a, idx)...] = val

# Depending on the type of the view we want to return either cartesian or linear index:

# ════════════════════════════════════════ NDSubArrayFast ════════════════════════════════════════ #

Base.IndexStyle(::NDSubArrayFast) = IndexLinear()

# ──── return a linear Index ───────────────────────────────────────────────────────────────────── #

@propagate_inbounds _getIndex(a::NDSubArrayFast{T,0}) where T = (
    a.offset1 + a.stride1 * firstindex(a)
)

@propagate_inbounds _getIndex(a::NDSubArrayFast{T,N}, i::Int) where {T,N} = (
    a.offset1 + a.stride1 * i
)

@propagate_inbounds _getIndex(a::NDSubArrayFast{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    _getIndex(a, from_cartesian_to_linear(size(a), I))
)

@propagate_inbounds _getIndex(a::NDSubArrayFast{T,N}, indices::NTuple{N,Any}) where {T,N} = (
    _getIndex(a, from_cartesian_to_linear(size(a), indices))
)

# ════════════════════════════════════════ NDSubArraySlow ════════════════════════════════════════ #
            
Base.IndexStyle(::NDSubArraySlow) = IndexCartesian()

# ──── return cartesian Index ──────────────────────────────────────────────────────────────────── #

# scalar
@propagate_inbounds _getIndex(a::NDSubArraySlow{T,0}) where T = (
    reindex(a.indices, firstindex(a))
)

# Linear index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T,N}, i::Int) where {T,N} = (
    _getIndex(a, from_linear_to_cartesian(size(a), i))
)

# Cartesian index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    reindex(a.indices, I)
)

# 'Normal' index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T,N}, indices::NTuple{N,Int}) where {T,N} = (
    reindex(a.indices, indices)
)

# ══════════════════════════════════════ internatl helpers ═══════════════════════════════════════ #

# ──── check when we reshape the parent ────────────────────────────────────────────────────────── #

# Both the parent and the future shape have the same dimensionality so don't reshape
maybe_reshape_parent(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A

# Otherwise if their dimension are different we reshape
maybe_reshape_parent(A::AbstractArray,          ::NTuple{N, Bool}) where {N}  = reshape(A, Val(N))

# ──── check view Indices ──────────────────────────────────────────────────────────────────────── #

check_view_indices(parent, indices) = check_view_indices(parent, index_ndims(indices...))

check_view_indices(parent::AbstractArray{T,N}, ::NTuple{N,Bool}) where {T,N} = nothing # valid case
check_view_indices(parent, ::NTuple{N,Bool}) where N = (
    @throw_error ArgumentError ERR_VIEW_INDICES_PARENT_DIM # invalid case
)

# ──── check for trivial view ──────────────────────────────────────────────────────────────────── #

is_trivial_view(v::NDSubArray, inds...) =
    all(enumerate(inds)) do (dim, idx)
        is_full_slice(v, idx, dim)
    end

# checks that all index are taken in the dimension
is_full_slice(A, index::Colon, dim::Int) = true
is_full_slice(A, index::Any, dim::Int) = false
is_full_slice(A, idx::AbstractRange, dim::Int) = (axes(A, dim) == first(idx):1:last(idx))

# ──── obtain axes from indices ────────────────────────────────────────────────────────────────── #

function _axes(a::NDSubArray{T,N}) where {T,N}
    collected_axes = []

    for arg in a.indices

        # push everything except scalar
        if arg isa AbstractArray
            for axis_range in axes(arg)
                push!(collected_axes, axis_range)
            end
        end
    end

    # For empty array
    if length(collected_axes) == 0 && N != 0
        return (Base.OneTo(0),)
    end
    return tuple(collected_axes...)
end
