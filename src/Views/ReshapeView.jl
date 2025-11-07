# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                         ReshapedArray42                                          #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct ReshapedArray42{T,N,P<:AbstractArray,Linear} <: AbstractNDArray{T,N}
    parent::P
    dims::NTuple{N,Int}
end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #

const ScalarReshapedArray{T,P} = ReshapedArray42{T,0,P}
const ReshapedArray42Fast{T,N,P<:AbstractArray} = ReshapedArray42{T,N,P,true}

# ──── constructor ─────────────────────────────────────────────────────────────────────────────── #

ReshapedArray42(parent::AbstractArray{T}, dims::NTuple{N,Int}, ::Val{Linear}) where {T,N,Linear} = (
    ReshapedArray42{T,N,typeof(parent), Linear}(parent, dims)
)

# ══════════════════════════════════ implements AbstractNDArray ══════════════════════════════════ #

# ──── Base functions ──────────────────────────────────────────────────────────────────────────── #

Base.parent(A::ReshapedArray42) = A.parent
Base.axes(A::ReshapedArray42) = map(Base.OneTo, size(A))

# ──── Internals ───────────────────────────────────────────────────────────────────────────────── #

@propagate_inbounds _getElement(a::ScalarReshapedArray{T}) where T = (
    __getElement(a)
)

@propagate_inbounds _setElement!(a::ScalarReshapedArray{T}, val) where T = (
    __setElement!(a, val)
)

# Linear index
@propagate_inbounds _getElement(a::ReshapedArray42{T,N}, i::Int) where {T,N} = (
    __getElement(a, i)
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T,N}, val, i::Int) where {T,N} = (
    __setElement!(a, val, i)
)

# Cartesian index
@propagate_inbounds _getElement(a::ReshapedArray42{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    __getElement(a, I)
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T,N}, v, I::AllCartesianIndex{N}) where {T,N} =
    __setElement!(a, v, I)

# 'Normal' index
@propagate_inbounds _getElement(a::ReshapedArray42{T,N}, indices::NTuple{N,Index}) where {T,N} = (
    __getElement(a, indices)
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T,N}, v, inds::NTuple{N,Index}) where {T,N} = (
    __setElement!(a, v, inds)
)

# ══════════════════════════════ functions for all ReshapedArray42 ═══════════════════════════════ #
            
# ──── core properties ─────────────────────────────────────────────────────────────────────────── #

Base.pointer(A::ReshapedArray42) = pointer(parent(A))
Base.size(A::ReshapedArray42) = A.dims
Base.length(A::ReshapedArray42) = length(parent(A))

# ──── similar function ────────────────────────────────────────────────────────────────────────── #

Base.similar(A::ReshapedArray42, eltype::Type, dims::Dims) = similar(parent(A), eltype, dims)

# Similar if parent is also a reshaped array
Base.similar(::Type{TA}, dims::Dims) where {T,N,P,TA<:ReshapedArray42{T,N,P}} = similar(P, dims)

# ──── getter/setter ───────────────────────────────────────────────────────────────────────────── #

__getElement(a::ReshapedArray42{T,0}) where T = parent(a)[_getIndex(a)...]
__getElement(a::ReshapedArray42{T,N}, I) where {T,N} = parent(a)[_getIndex(a, I)...]

__setElement!(a::ReshapedArray42{T,0}, val) where T = parent(a)[_getIndex(a)...] = val
__setElement!(a::ReshapedArray42{T,N}, val, I) where {T,N} = parent(a)[_getIndex(a, I)...] = val

# ═══════════════════════════════════════ ReshapedArrayLF ════════════════════════════════════════ #

Base.IndexStyle(::Type{<:ReshapedArray42Fast}) = IndexLinear()

# ──── return a linear index ───────────────────────────────────────────────────────────────────── #
            
@propagate_inbounds _getIndex(A::ReshapedArray42Fast{T,0}) where T = 1

# Linear index
@propagate_inbounds _getIndex(A::ReshapedArray42Fast{T,N}, i::Int) where {T,N} = (
    i - firstindex(A) + firstindex(parent(A))
)

# Cartesian index
@propagate_inbounds _getIndex(A::ReshapedArray42Fast{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    _getIndex(A, LinearIndex(A, I))
)

# 'Normal' index
@propagate_inbounds _getIndex(A::ReshapedArray42Fast{T,N}, indices::NTuple{N,Any}) where {T,N} = (
    _getIndex(A, LinearIndex(A, indices))
)

# ═══════════════════════════════════ ReshapedArray non-linear ═══════════════════════════════════ #
            
Base.IndexStyle(::Type{<:ReshapedArray42{T,N}}) where {T,N} = IndexCartesian()

# ──── return a cartesian index ────────────────────────────────────────────────────────────────── #

@propagate_inbounds _getIndex(A::ScalarReshapedArray{T}) where T = (
    _getIndex(A, ntuple(_ -> 1, Base.ndims(parent(A))))
)

# TODO: see if we can directly do from linear to cartesian
@propagate_inbounds _getIndex(A::ReshapedArray42{T,N}, i::Int) where {T,N} = (
    _getIndex(A, CartesianIndex42(A, i))
)

# Cartesian index
@propagate_inbounds _getIndex(A::ReshapedArray42{T,N}, I::AllCartesianIndex{N}) where {T,N} = (
    i = LinearIndex(A, I);
    return CartesianIndex42(parent(A), i)
)

# 'Normal' index
@propagate_inbounds _getIndex(A::ReshapedArray42{T,N}, indices::NTuple{N,Any}) where {T,N} = (
    i = LinearIndex(A, indices);
    return CartesianIndex42(parent(A), i)
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                         reshape function                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# ════════════════════════════════════ reshape to n dimension ════════════════════════════════════ #

# ──── reshape function ────────────────────────────────────────────────────────────────────────── #
            
reshape(parent::AbstractArray{T,N}, ::Val{N}) where {T,N} = parent

reshape(parent::AbstractArray, ndims::Val{N}) where N = (
    reshape(parent, view_dims(ndims, axes(parent)))
)

# ──── obtain target dimension ─────────────────────────────────────────────────────────────────── #

# view dims return the shape that will be used to reshape the parent

# target is scalar -> return an empty array
view_dims(::Val{0}, inds::Tuple) = ()

# target dims equal source dims -> simply return the indices
view_dims(::Val{N}, inds::Tuple{N,Any}) where N = inds


# Helper function to compute the total length of a tuple of indices/axes
function trailing_length(inds)
    total_length = 1
    for ax in inds
        total_length *= length(ax)
    end
    return total_length
end

function view_dims(::Val{N}, inds::NTuple{M,Any}) where {M,N}
    # (target dims > source dims): Append (N-M) OneTo(1) axes
    if N > M
        new_dims = collect(Any, inds)
        for i in 1:(N-M)
            push!(new_dims, Base.OneTo(1))
        end
        return tuple(new_dims...)
    end

    # N < M (target dims < source dims): Collapse trailing dimensions
    if N < M
        # Take the first N-1 dimensions as-is
        first_dims = (inds[i] for i in 1:N-1)

        # Collapse the remaining trailing dimensions
        collapsed_length = trailing_length((inds[i] for i in N:M))

        return (first_dims..., Base.OneTo(collapsed_length))
    end

    # Should not be reached
    error("Invalid logic in view_dims")
end

# ═══════════════════════════════════════ reshape to dims ════════════════════════════════════════ #
            
# ──── reshape with dims argument ──────────────────────────────────────────────────────────────── #

# Reshape with integer arguments
reshape(parent::AbstractArray, dims::Integer...) = reshape(parent, to_sizes(dims))

# Reshape with tuple of integers or ranges
reshape(parent::AbstractArray, dims::Tuple) = reshape(parent, to_sizes(dims))

# little helper
to_sizes(dims) = map(d -> d isa AbstractRange ? length(d) : Int(d), dims)

# ──── catch all reshape function  ─────────────────────────────────────────────────────────────── #

# Reshape for reshaped array
reshape(parent::ReshapedArray42, dims::Dims) = reshape(parent.parent, dims)

# Final reshape using Dims
function reshape(parent::AbstractArray, dims::Dims)
    n = length(parent)
    same_length = prod(dims) == n
    if (same_length)
        return _reshape((parent, IndexStyle(parent)), dims)
    end
    s = "parent has $n elements, which is incompatible with dimensions $dims $(prod(dims)) elements"
    throw(DimensionMismatch(s))
end

# scalar and linearIndex
function _reshape(p::Tuple{AbstractArray,IndexLinear}, dims::Dims)
    ReshapedArray42(p[1], dims, Val(true))
end

# CartesianIndex
function _reshape(p::Tuple{AbstractArray,Base.IndexStyle}, dims::Dims)
    ReshapedArray42(p[1], dims, Val(false))
end

# ══════════════════════════════════════ dropdims function ═══════════════════════════════════════ #

# Drop singleton dimension like (2, 2, 1) to (2, 2)
function Base.dropdims(c::AbstractArray; dims::Union{Nothing,Integer,Tuple}=nothing)

    # If none are specified drop all dead dimension
    if dims === nothing
        dims_to_drop = Tuple(i for (i, d) in enumerate(size(c)) if d == 1)

        # If some are specified we verify that their value is indeed 1
    else
        dims_tuple = isa(dims, Integer) ? (dims,) : dims
        for d in dims_tuple
            if size(c, d) != 1
                s = "Dimension $d asked for dropdims but size is $(size(c, d)), not 1."
                throw(DimensionMismatch(s))
            end
        end
        dims_to_drop = Tuple(dims_tuple)
    end

    if isempty(dims_to_drop)
        return c
    end

    parent_dims = size(c)
    new_dims = filter(d -> d != 1, parent_dims)
    return reshape(c, new_dims)
end

# Drop dims tuple version
function Base.dropdims(dims::Dims)

    N = length(dims)

    for i in N:-1:1
        if (dims[i] == 1)
            N -= 1
        end
    end

    return ntuple(i -> dims[i], N)
end



# ════════════════════════════════════ reshaped Array strides ════════════════════════════════════ #

# Computing the strides of a reshaped array is subtle. The process works roughly as follows:
#
# 1. First, check whether the reshaped array is contiguous in memory. 
#    In that case, computing strides is trivial from the shape alone.
#
# 2. If the reshaped array does not appear contiguous, we next attempt to
#    merge adjacent dimensions whose strides indicate they are contiguous
#    in memory. This handles cases where the parent array is contiguous,
#    but the reshape shape breaks the simple contiguity pattern.
#
# 3. Once we encounter a boundary where contiguity fails, we have identified
#    a block of truly contiguous dimensions. We record that merged block.
#
# 4. We continue merging across the remaining dimensions, forming as many
#    contiguous stride-blocks as possible.
#
# 5. If merging completes cleanly, we can derive the new strides. Otherwise,
#    the reshaped array is not representable as a standard strided layout
#    (i.e., it is not a true strided array), and we must fall back to a
#    more general handling path.

# ──── Determine contingous array ──────────────────────────────────────────────────────────────── #

# Determine which type is contingous in memory

_checkcontiguous(::Type{Bool}, A::NDArray) = true
_checkcontiguous(::Type{Bool}, A::NDSubArrayFast) = true
_checkcontiguous(::Type{Bool}, A::NDSubArraySlow) = false
_checkcontiguous(::Type{Bool}, A::AbstractArray) = false
_checkcontiguous(::Type{Bool}, A::ReshapedArray42) = _checkcontiguous(Bool, parent(A))

_checkcontiguous(::Type{Bool}, A::Base.DenseArray) = true
_checkcontiguous(::Type{Bool}, A::Base.ReshapedArray) = _checkcontiguous(Bool, parent(A))
_checkcontiguous(::Type{Bool}, A::Base.FastContiguousSubArray) = _checkcontiguous(Bool, parent(A))

# ──── compute strides ─────────────────────────────────────────────────────────────────────────── #

# The parent is a scalar
Base.strides(a::ReshapedArray42{DType,N,P}) where {DType,N,P<:AbstractArray{DType,0}} = (
    ntuple(_ -> 1, N)
)

# Return the strides of the reshaped array
function Base.strides(a::ReshapedArray42)
    # If contingous directly return the strides from the size
    _checkcontiguous(Bool, a) && return compute_strides(size(a))

    # Else we try to merge the dimensin in a 'lazy' way
    parent_size, parent_stride = size(a.parent), strides(a.parent)
    merged_size, merged_strides, n = merge_adjacent_dim(parent_size, parent_stride)

    # If the number of dimensions merged 'n' is the same as the parent, then the merged dimensions 
    # are contingous
    if (n == ndims(a.parent))
        return compute_strides(tuple(merged_strides...), size(a))
    end
    # Otherwise there is still more work to do
    return _reshaped_strides(size(a), 1, merged_size, merged_strides, n, parent_size, parent_stride)
end

# ──── merge adjacent dimension ────────────────────────────────────────────────────────────────── #
            
merge_adjacent_dim(::Dims{0}, ::Dims{0}) = 1, 1, 0

merge_adjacent_dim(parent_sizes::Dims{1}, parent_strides::Dims{1}) = (
    parent_sizes[1], parent_strides[1], 1
)

function merge_adjacent_dim(parent_sizes::Dims{N}, parent_strides::Dims{N}, n::Int=1) where {N}
    # Init: size and stride of dimension n
    sizeₙ, strideₙ = parent_sizes[n], parent_strides[n]

    # n: Index of the actual dimension we try to merge
    while n < N
        # Init: size and stride of the next dimension 'n+1'
        sizeₙ₊₁, strideₙ₊₁ = parent_sizes[n+1], parent_strides[n+1]

        # Case 1: The actual dimension is of size 1, it's ignored because it don't break continguity
        if sizeₙ == 1
            sizeₙ, strideₙ = sizeₙ₊₁, strideₙ₊₁
            # Case 2: One of the following conditions is met so it's still contiguous
            # a) strideₙ₊₁ == strideₙ * sizeₙ : memory layout remains contiguous: 
            #   next dimension directly follows the previous in memory
            # b) sizeₙ₊₁ == 1: the next dimension's size is 1, so it can be merged
        elseif strideₙ₊₁ == strideₙ * sizeₙ || sizeₙ₊₁ == 1
            sizeₙ *= sizeₙ₊₁
            # Case 3: The next dimension is not contingous so exit the loop
        else
            break
        end
        # Préparer la prochaine itération
        n += 1
    end
    # sizeₙ:   Total size of the merged block
    # strideₙ: Start stride of the merged block
    # n: Index of the first non merged dimension
    return sizeₙ, strideₙ, n
end

# ──── merged the remaining dimension ──────────────────────────────────────────────────────────── #

# Base case: called when all dimensions of the reshaped array view (`size`) have been processed.
function _reshaped_strides(::Dims{0}, reshaped::Int, merged_size::Int, ::Int, ::Int, ::Dims, ::Dims)
    # If the size of the merged dim match the parent's, then the view successfully consumed
    # the parent's memory
    reshaped == merged_size && return ()

    # Otherwise, the reshaped array cannot be represented by simple linear strides 
    # (it breaks contiguity rules), so we throw an error.
    throw(ArgumentError("Input is not strided."))
end

# The recursive function that calculates the stride for all the dimensions 
function _reshaped_strides(size::Dims, reshaped::Int, merged_size::Int, 
    merged_strides::Int, n::Int, apsz::Dims, apst::Dims)

    # 1. Calculate the stride for the current dimension
    st = reshaped * merged_strides

    # 2. Update the accumulated size to include the current dimension.
    reshaped = reshaped * size[1]

    # 3. Check if the conditions for merging the actual and next dimension is met
    if length(size) > 1 && reshaped == merged_size && size[2] != 1
        merged_size, merged_strides, n = merge_adjacent_dim(apsz, apst, n + 1)

        # Reset the accumulated view size since we are starting a new block in the parent.
        reshaped = 1
    end

    # 4. Recursively calculate the strides for the remaining dimensions of the view.
    sts = _reshaped_strides(Base.tail(size), reshaped, merged_size, merged_strides, n, apsz, apst)

    # 5. Return the stride for the current dimension (`st`) concatenated with 
    #    the rest of the strides (`sts`).
    return (st, sts...)
end


# ──── iteration ───────────────────────────────────────────────────────────────────────────────── #

# Index (simple wrapper with default constructor)
struct ReshapedIndex{T}
    parentindex::T
end

# Iterator for the reshaped array
struct ReshapedArray42Iterator{I}
    iter::I
end

# Constructor
ReshapedArray42Iterator(A::ReshapedArray42) = begin
    P = parent(A)
    iter = eachindex(P)
    ReshapedArray42Iterator{typeof(iter)}(iter)
end

# Basic functions: 
Base.length(R::ReshapedArray42Iterator) = (
    length(R.iter)
)

Base.eltype(::Type{<:ReshapedArray42Iterator{I}}) where {I} = (
    @isdefined(I) ? ReshapedIndex{eltype(I)} : Any
)

# Tells that when iterating on a ReshapedArray42 we use the right iterator 
Base.eachindex(A::ReshapedArray42) = (
    ReshapedArray42Iterator(A)
)

# Returns the iterator contained in the parentindex
Base.lastindex(a::ReshapedArray42{T,N}) where {T,N} = last(eachindex(a)).parentindex
Base.firstindex(a::ReshapedArray42{T,N}) where {T,N} = first(eachindex(a)).parentindex

# Iteration function: 
@inline function Base.iterate(R::ReshapedArray42Iterator, i...)
    state = iterate(R.iter, i...)
    state === nothing && return nothing

    item, inext = state
    return (ReshapedIndex(item), inext)
end

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.getindex(A::ReshapedArray42, index::ReshapedIndex) = (
    @boundscheck checkbounds(parent(A), index.parentindex);
    @inbounds parent(A)[index.parentindex]
)

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.setindex!(A::ReshapedArray42, val, index::ReshapedIndex) = (
    @boundscheck checkbounds(parent(A), index.parentindex);
    @inbounds parent(A)[index.parentindex] = val
)

# ──── copy functions ──────────────────────────────────────────────────────────────────────────── #
            
# We override those case to avoid using the ReshapedArray42Iterator in other AbstractNDArray

function Base.copy!(dest::AbstractNDArray, src::ReshapedArray42Fast)
    @boundscheck size(dest) == size(src)
    for idx in 1:length(src)
        dest[idx] = src[idx]
    end
    return dest
end

function Base.copy!(dest::AbstractNDArray, src::ReshapedArray42)
    @boundscheck size(dest) == size(src)
    for idx in CartesianIndices42(src)
        dest[idx] = src[idx]
    end
    return dest
end

# ──── print function ──────────────────────────────────────────────────────────────────────────── #
            
function Base.show(io::IO, A::ReshapedArray42{T, N}) where {T, N}
    println(io, typeof(A), " with dims: ", A.dims)
    if N == 0
        println(A[])
    elseif N == 1
        Base.show_vector(io, A)
    else
        Base.print_array(io, A)
    end
end

# ──── reshaped range error ────────────────────────────────────────────────────────────────────── #
            
const ReshapedRange{T,N,A<:AbstractRange} = ReshapedArray42{T,N,A,Tuple{}}

Base.setindex!(A::ReshapedRange, val, index::Int) = 
    error("indexed assignment fails for a reshaped range; consider calling collect")

Base.setindex!(A::ReshapedRange{T,N}, val, indices::Vararg{Int,N}) where {T,N} =
    error("indexed assignment fails for a reshaped range; consider calling collect")

Base.setindex!(A::ReshapedRange, val, index::ReshapedIndex) = 
    error("indexed assignment fails for a reshaped range; consider calling collect")