using Pkg

Pkg.add("Test")


using Base.MultiplicativeInverses: SignedMultiplicativeInverse
using Base:  IndexStyle, IndexLinear, tail, front, Dims # Importations Julia standard
using Test

struct ReshapedArray42{T,N,P<:AbstractArray,MI<:Tuple{Vararg{SignedMultiplicativeInverse{Int}}}} <: AbstractArray{T,N}
    parent::P
    dims::NTuple{N,Int}
    mi::MI
end

# Alias for linear reshaped
const ReshapedArray42LF{T,N,P<:AbstractArray} = ReshapedArray42{T,N,P,Tuple{}}

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Constructor:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

ReshapedArray42(parent::AbstractArray{T}, dims::NTuple{N,Int}, mi) where {T,N} = ReshapedArray42{T,N,typeof(parent),typeof(mi)}(parent, dims, mi)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                         Implementation of abstract methods:                                   #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.IndexStyle(::Type{<:ReshapedArray42LF}) = IndexLinear()

# Parent
Base.parent(A::ReshapedArray42) = A.parent

# Size and axes
Base.size(A::ReshapedArray42)   = A.dims
Base.axes(A::ReshapedArray42)   = map(Base.OneTo, size(A))
Base.length(A::ReshapedArray42) = length(parent(A))

# Similar
Base.similar(A::ReshapedArray42, eltype::Type, dims::Dims) = similar(parent(A), eltype, dims)
Base.similar(::Type{TA}, dims::Dims) where {T,N,P,TA<:ReshapedArray42{T,N,P}} = similar(P, dims)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         Iteration :                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Index (simple wrapper with default constructor)
struct ReshapedIndex{T}
    parentindex::T
end

# Iterator for the reshaped array
struct ReshapedArray42Iterator{I,M}
    iter::I
    mi::NTuple{M, SignedMultiplicativeInverse{Int}}
end

# Constructor
ReshapedArray42Iterator(A::ReshapedArray42) = begin
    P = parent(A)
    iter = eachindex(P)
    ReshapedArray42Iterator{typeof(iter), length(A.mi)}(iter, A.mi) 
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

# Iteration function: 
@inline function iterate(R::ReshapedArray42Iterator, i...)
    item, inext = iterate(R.iter, i...)
    ReshapedIndex(item), inext
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Getindex :                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Reshaped LF LinearIndex    
@inline Base.getindex(A::ReshapedArray42LF, index::Int) = (
    @boundscheck checkbounds(A, index);
    # Convert `A`'s linear index to the corresponding linear index in the parent array
    indexparent = index - firstindex(A) + firstindex(parent(A));
    @inbounds parent(A)[indexparent]
)

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.getindex(A::ReshapedArray42, index::ReshapedIndex) = (
    @boundscheck checkbounds(parent(A), index.parentindex);
    @inbounds parent(A)[index.parentindex]
)

# Reshaped Array with 'Normal' indices
@inline Base.getindex(A::ReshapedArray42{T,N}, indices::Vararg{Int,N}) where {T,N} = (
    @boundscheck checkbounds(A, indices...);
    _unsafe_getindex(A, indices...)
)

@inline function _unsafe_getindex(A::ReshapedArray42{T,N}, indices::Vararg{Int,N}) where {T,N}
    parent_axes = axes(A.parent)

    # Convert the index relative to the reshaped array into a linear index 'i',
    # then back to Cartesian indices 'I' in the parent to handle offsets and reshaping correctly.
    i = _offset_index(_cartesian_to_linear(size(A), indices), parent_axes)
    I = _unravel_index(parent_axes, A.mi, i)
    return @inbounds parent(A)[I...]
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Setindex :                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Reshaped LF LinearIndex
@inline Base.setindex!(A::ReshapedArray42LF, val, index::Int) = (
    @boundscheck checkbounds(A, index);
    # Convert `A`'s linear index to the corresponding linear index in the parent array
    indexparent = index - firstindex(A) + firstindex(parent(A));
    @inbounds parent(A)[indexparent] = val
)

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.setindex!(A::ReshapedArray42, val, index::ReshapedIndex) = (
    @boundscheck checkbounds(parent(A), index.parentindex);
    @inbounds parent(A)[index.parentindex] = val
)

# Reshaped Array with 'Normal' indices
@inline Base.setindex!(A::ReshapedArray42{T,N}, val, indices::Vararg{Int,N}) where {T,N} = (
    @boundscheck checkbounds(A, indices...);
    _unsafe_setindex!(A, val, indices...)
)

@inline function _unsafe_setindex!(A::ReshapedArray42{T,N}, val, indices::Vararg{Int,N}) where {T,N}
    parent_axes = axes(A.parent)

    # Convert the index relative to the reshaped array into a linear index 'i',
    # then back to Cartesian indices 'I' in the parent to handle offsets and reshaping correctly.
    i = _offset_index(_cartesian_to_linear(size(A), indices), parent_axes)
    I = _unravel_index(parent_axes, A.mi, i)
    return @inbounds parent(A)[I...] = val
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Reshape :                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Matches reshape(A, 2, 3)
reshape(parent::AbstractArray, dims::Integer...) = reshape(parent, map(Int, dims))

# Matches reshape(A, (2, 3))
reshape(parent::AbstractArray, dims::Tuple{Integer, Vararg{Integer}}) = reshape(parent, map(Int, dims))

function reshape(parent::AbstractArray, dims::Dims)
    n = prod(Base.size(parent))
    prod(dims) == n || throw( DimensionMismatch( "parent has $n elements, which is incompatible with dimensions $dims ($(prod(dims)) elements)"))
    _reshape((parent, IndexStyle(parent)), dims)
end

# Scalar
function _reshape(p::Tuple{AbstractArray{<:Any,0},Base.IndexStyle}, dims::Dims)
    ReshapedArray42(p[1], dims, ())
end

# LinearIndex
function _reshape(p::Tuple{AbstractArray,IndexLinear}, dims::Dims)
    ReshapedArray42(p[1], dims, ()) 
end

# CartesianIndex
function _reshape(p::Tuple{AbstractArray,Base.IndexStyle}, dims::Dims)
    parent = p[1]
    parent_dims = Base.front(size(parent))                 # all dimensions except the last one
    safe_sizes = map(s -> max(1, Int(s)), parent_dims)     # avoid 0 and neg values
    mi = map(Base.SignedMultiplicativeInverse, safe_sizes)
    ReshapedArray42(parent, dims, mi)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Strides :                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function strides(a::ReshapedArray42)
    # If contingous directly return the strides from the size
    _checkcontiguous(Bool, a) && return _strides_from_size(1, size(a)...)

    # Else we try to merge the dimensin in a 'lazy' way
    parent_size, parent_stride     = size(a.parent), strides(a.parent)
    merged_size, merged_strides, n = merge_adjacent_dim(parent_size, parent_stride)

    # If the number of dimensions merged 'n' is the same as the parent, then the merged dimensions are contingous
    if (n == ndims(a.parent))
        return _strides_from_size(merged_strides, size(a)...)
    end
    return _reshaped_strides(size(a), 1, merged_size, merged_strides, n, parent_size, parent_stride)
end

# compute strides from size:
# --------------------------

# Recursively compute strides for each dimension given starting stride `s` and sizes `d, sz...`.
@inline _strides_from_size(s, d, sz...) = (s, _strides_from_size(s * d, sz...)...)
_strides_from_size(s, d) = (s,)
_strides_from_size(s) = ()


# merge contingous dimensions:
# ---------------------------

merge_adjacent_dim(::Dims{0}, ::Dims{0}) = 1, 1, 0
merge_adjacent_dim(parent_sizes::Dims{1}, parent_strides::Dims{1}) = parent_sizes[1], parent_strides[1], 1

function merge_adjacent_dim(parent_sizes::Dims{N}, parent_strides::Dims{N}, n::Int = 1) where {N}
    # Init: size and stride of dimension n
    sizeₙ, strideₙ = parent_sizes[n], parent_strides[n]
    
    # n: Index of the actual dimension we try to merge
    while n < N
        # Init: size and stride of the next dimension 'n+1'
        sizeₙ₊₁, strideₙ₊₁ = parent_sizes[n+1], parent_strides[n+1]
         
        # Case 1: The actual dimension is of size 1, it is ignored because it do not break continguity
        if sizeₙ == 1
            sizeₙ, strideₙ = sizeₙ₊₁, strideₙ₊₁
        # Case 2: One of the following conditions is met so it's still contiguous
        # a) strideₙ₊₁ == strideₙ * sizeₙ : memory layout remains contiguous: next dimension directly follows the previous in memory
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

# compute the strides of the remaining non-merged block:
# -----------------------------------------------------

# Base case: called when all dimensions of the reshaped array view (`size`) have been processed.
function _reshaped_strides(::Dims{0},reshaped::Int, merged_size::Int, ::Int, ::Int, ::Dims, ::Dims)
    # If the size of the merged dim match the parent's, then the view successfully consumed the parent's memory
    reshaped == merged_size && return ()

    # Otherwise, the reshaped array cannot be represented by simple linear strides (it breaks contiguity rules), so we throw an error.
    throw(ArgumentError("Input is not strided."))
end

# The recursive function that calculates the stride for all the dimensions 
function _reshaped_strides(size::Dims, reshaped::Int, merged_size::Int, merged_strides::Int, n::Int, apsz::Dims, apst::Dims)

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
    sts = _reshaped_strides(tail(size), reshaped, merged_size, merged_strides, n, apsz, apst)
    
    # 5. Return the stride for the current dimension (`st`) concatenated with the rest of the strides (`sts`).
    return (st, sts...)
end

#_checkcontiguous(::Type{Bool}, A::NDArray) = true
#_checkcontiguous(::Type{Bool}, A::Array) = true
_checkcontiguous(::Type{Bool}, A::AbstractArray) = false
#_checkcontiguous(::Type{Bool}, A::CompoundView) = false
#_checkcontiguous(::Type{Bool}, A::AbstractNDView) = _checkcontiguous(Bool, parent(A))
_checkcontiguous(::Type{Bool}, A::ReshapedArray42) =  _checkcontiguous(Bool, parent(A))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Index helpers   :                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Corrects the linear index for arrays whose first index tuple (like (1,1,…)) is offset, as in some views.
_offset_index(i::Integer, axs::Tuple) = i
_offset_index(i::Integer, axs::Tuple{<:AbstractUnitRange}) = i + first(axs[1]) - 1

# The goal of this function is to convert a single linear index into the multi-dimensional indices corresponding to the parent array.
@inline _unravel_index(parent_axes, parent_mi, linear_index::Int) = _compute_parent_cartesian(parent_axes, parent_mi, linear_index - 1)

# Base case: handles the last dimension of the parent. Since no multiplicative inverses remain, this final dim is the remainder index.
@inline _compute_parent_cartesian(parent_axes, ::Tuple{}, remainder_index) = (remainder_index + first(parent_axes[end]),)


# Recursive case: converts a portion of a linear index into Cartesian indices for the parent array.
# mi:               precomputed stride-like factors (historically called multiplicative inverses) 
#                   used to efficiently extract each dimension’s index
# dim_ranges:       the index ranges for each parent dimension
# remainder_index:  the part of the linear index still to process
@inline function _compute_parent_cartesian(dim_ranges, mi, remainder_index)
    # Step 1: Obtain the index of this dimension and the remainder of the division
    dim_remainder, dim_index = divrem(remainder_index, mi[1])

    # Step 2: Build the Cartesian index tuple
    # - Add the axis offset to dim_index to get the actual index in the parent array
    # - Recursively compute indices for the remaining dimensions using dim_remainder
    return ( 
        dim_index + first(dim_ranges[1]), 
        _compute_parent_cartesian(tail(dim_ranges), tail(mi), dim_remainder)...
    )
end

Base.@propagate_inbounds function _cartesian_to_linear(dims::Dims, indices::NTuple{N, Int}) where {T, N}    
    offset = 0 # Offset 0-based
    s = 1      # Stride courant
    
    @inbounds for i in 1:N
        # (indices[i] - 1) * s est l'offset contributif de la dimension i
        offset += (indices[i] - 1) * s 
        
        # Le prochain stride est (stride courant * taille de la dimension)
        s *= dims[i]
    end
    
    # L'indice linéaire (1-based) est l'offset + 1
    return offset + 1
end

# -----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES POUR LE TEST (pour simplifier la création)
# -----------------------------------------------------------------------------

# Simule la logique de Base.__reshape pour créer une instance avec MI
function create_reshaped_array(P::AbstractArray, DIMS::Dims)
    # Cas Indexation Cartésienne (avec MI)
    if IndexStyle(P) isa Base.IndexCartesian 
        szs = Base.front(size(P))
        szs1 = map(s -> max(1, Int(s)), szs)
        mi = map(SignedMultiplicativeInverse, szs1)
        return ReshapedArray42(P, DIMS, mi)
    # Cas Indexation Linéaire (sans MI)
    else
        return ReshapedArray42(P, DIMS, ())
    end
end


const ReshapedRange{T,N,A<:AbstractRange} = ReshapedArray42{T,N,A,Tuple{}}

Base.setindex!(A::ReshapedRange, val, index::Int) = error("indexed assignment fails for a reshaped range; consider calling collect")
Base.setindex!(A::ReshapedRange{T,N}, val, indices::Vararg{Int,N}) where {T,N} = error("indexed assignment fails for a reshaped range; consider calling collect")
Base.setindex!(A::ReshapedRange, val, index::ReshapedIndex) = error("indexed assignment fails for a reshaped range; consider calling collect")





# firstindex(a::AbstractArray) = (@inline; first(eachindex(IndexLinear(), a)))
# firstindex(a, d) = (@inline; first(axes(a, d)))
#lastindex(a::AbstractArray) = (@inline; last(eachindex(IndexLinear(), a)))
#lastindex(a, d) = (@inline; last(axes(a, d)))