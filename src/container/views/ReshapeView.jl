# using Pkg

# Pkg.add("Test")


using Base.MultiplicativeInverses: SignedMultiplicativeInverse
using Base:  IndexStyle, IndexLinear, tail, front, Dims # Importations Julia standard
#using Test

struct ReshapedArray42{T,N,P<:AbstractArray,MI<:Tuple{Vararg{SignedMultiplicativeInverse{Int}}}} <: AbstractNDArray{T,N}
    parent::P
    dims::NTuple{N,Int}
    mi::MI
end

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

# Parent
Base.parent(A::ReshapedArray42) = A.parent
Base.pointer(A::ReshapedArray42) = pointer(parent(A))

# Size and axes
Base.size(A::ReshapedArray42)   = A.dims
Base.axes(A::ReshapedArray42)   = map(Base.OneTo, size(A))
Base.length(A::ReshapedArray42) = length(parent(A))

# Similar
Base.similar(A::ReshapedArray42, eltype::Type, dims::Dims) = similar(parent(A), eltype, dims)
Base.similar(::Type{TA}, dims::Dims) where {T,N,P,TA<:ReshapedArray42{T,N,P}} = similar(P, dims)


# function Base.view(a::ReshapedArray42{T,N}, inds...) where {T,N}
#     println("We want A[$inds] where A is: ")
#     println(a)
#     println("-------------- Creation View from ReshapedArray42 (indices: $inds) ---------------- ")
#     # check bounds
#   # check bounds
#     J = to_indices(a, inds)

#     @boundscheck checkbounds(a, J...)
    
#     # drop dimension
#     J_2 =  drop_singleton_dimension(J, ndims(a))
    
#     # resize parent if needed
#     reshaped_parent = maybe_reshape_parent(a, Base.index_ndims(J_2...))
#     size_before, size_now = size(a), size(reshaped_parent)
#     println("J: $J")
#     println("J': $J_2")

#     # print the content
#     println("content: ", ((size_before != size_now) ? " (reshaped from $size_before to $size_now) " : " " ) * string(reshaped_parent) )

#     println("elements befoer beginning the view: reshaped_parent: ndimsA: $(ndims(a)) $reshaped_parent, index_ndims: $(Base.index_ndims(J_2...))")
#     # create the view
#     V =  create_view(reshaped_parent, J_2...)
#     println("offset1: ", V.offset1)
#     println("firstindex: ", firstindex(V))
#     println("lastindex: ", lastindex(V))
#     println("axes: ", axes(V))
#         println("indexStyle: ", IndexStyle(V))

#      println("------------------------------------------------------------------------- \n\n")
#     return V
# end

function Base.strides(a::ReshapedArray42{DType,N, P}) where {DType,N, P <: AbstractArray{DType, 0}}
    return ntuple(_ -> 1, N)
end

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
@inline function Base.iterate(R::ReshapedArray42Iterator, i...)
    item, inext = iterate(R.iter, i...)
    ReshapedIndex(item), inext
end

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.getindex(A::ReshapedArray42, index::ReshapedIndex) = (
    @boundscheck checkbounds(A, index.parentindex);
    @inbounds parent(A)[index.parentindex]
)

# Reshaped Array ReshapedIndex ( used in iteration )
@inline Base.setindex!(A::ReshapedArray42, val, index::ReshapedIndex) = (
    @boundscheck checkbounds(parent(A), index.parentindex);
    @inbounds parent(A)[index.parentindex] = val
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get and set element:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

@propagate_inbounds _getElement(a::ReshapedArray42{T, 0}) where {T} = (
    parent = Base.parent(a); 
    parent[_getIndex(a)...]
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T, 0}, val::T) where {T} = (
    parent = Base.parent(a);
    parent[_getIndex(a)...] = val
)

# Linear index
@propagate_inbounds _getElement(a::ReshapedArray42{T, N}, i::Int) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, i)...]
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T, N}, val::T, i::Int) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, i)...] = val
)


# Cartesian index
@propagate_inbounds _getElement(a::ReshapedArray42{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, I)...]
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T, N}, val::T, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, I)...] = val
)

# 'Normal' index
@propagate_inbounds _getElement(a::ReshapedArray42{T, N}, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a); 
    parent[_getIndex(a, indices)...]
)

@propagate_inbounds _setElement!(a::ReshapedArray42{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, indices)...] = val
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Linear Index:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Alias for linear reshaped
const ReshapedArray42LF{T,N,P<:AbstractArray} = ReshapedArray42{T,N,P,Tuple{}}

Base.IndexStyle(::Type{<:ReshapedArray42LF}) = IndexLinear()

# Scalar
@propagate_inbounds _getIndex(A::ReshapedArray42LF{T, 0}) where T = 1

# Linear index
@propagate_inbounds _getIndex(A::ReshapedArray42LF{T, N}, i::Int) where {T, N} = i - firstindex(A) + firstindex(parent(A))

# Cartesian index
@propagate_inbounds _getIndex(A::ReshapedArray42LF{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = _getIndex(A, offset(A, I))

# 'Normal' index
@propagate_inbounds _getIndex(A::ReshapedArray42LF{T, N}, indices::NTuple{N, Any}) where {T, N} = _getIndex(A, offset(A, indices))
``
# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Cartesian Index:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.IndexStyle(::Type{<:ReshapedArray42{T, N}}) where {T, N} = IndexCartesian()

# scalar
@propagate_inbounds _getIndex(A::ReshapedArray42{T, 0}) where T = _getIndex(A, ntuple(_ -> 1, Base.ndims(parent(A))))

# Linear index
@propagate_inbounds _getIndex(A::ReshapedArray42{T, N}, i::Int) where {T, N} = _getIndex(A, fromLinearToCartesian(i, A))

# Cartesian index
@propagate_inbounds _getIndex(A::ReshapedArray42{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = _getIndex(A, Tuple(I))

# 'Normal' index
@propagate_inbounds function _getIndex(A::ReshapedArray42{T, N}, indices::NTuple{N, Any}) where {T, N}
    parent_axes = axes(A.parent)

    # Convert the index relative to the reshaped array into a linear index 'i',
    # then back to Cartesian indices 'I' in the parent to handle offsets and reshaping correctly.
    i = _offset_index(_cartesian_to_linear(size(A), indices), parent_axes)
    I = _unravel_index(parent_axes, A.mi, i)
    return I
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

reshape(parent::AbstractArray{T,N}, ndims::Val{N}) where {T,N} = parent

function reshape(parent::AbstractArray, dims::Tuple{Vararg{Union{AbstractRange, Integer}}}) where N
    reshape(parent, map(idx -> idx isa AbstractRange ? length(idx) : idx, dims))
end

reshape(parent::AbstractArray{T,N}, ndims::Val{N}) where {T,N} = parent

function reshape(parent::AbstractArray, ndims::Val{N}) where N
    reshape(parent, rdims(Val(N), axes(parent)))
end

# Move elements from inds to out until out reaches the desired
# dimensionality N, either filling with OneTo(1) or collapsing the
# product of trailing dims into the last element
rdims_trailing(l, inds...) = length(l) * rdims_trailing(inds...)
rdims_trailing(l) = length(l)
rdims(out::Val{N}, inds::Tuple) where {N} = rdims(ntuple(Returns(Base.OneTo(1)), Val(N)), inds)
rdims(out::Tuple{}, inds::Tuple{}) = () # N == 0, M == 0
rdims(out::Tuple{}, inds::Tuple{Any}) = ()
rdims(out::Tuple{}, inds::NTuple{M,Any}) where {M} = ()
rdims(out::Tuple{Any}, inds::Tuple{}) = out # N == 1, M == 0
rdims(out::NTuple{N,Any}, inds::Tuple{}) where {N} = out # N > 1, M == 0
rdims(out::Tuple{Any}, inds::Tuple{Any}) = inds # N == 1, M == 1
rdims(out::Tuple{Any}, inds::NTuple{M,Any}) where {M} = (Base.oneto(rdims_trailing(inds...)),) # N == 1, M > 1
rdims(out::NTuple{N,Any}, inds::NTuple{N,Any}) where {N} = inds # N > 1, M == N
rdims(out::NTuple{N,Any}, inds::NTuple{M,Any}) where {N,M} = (first(inds), rdims(tail(out), tail(inds))...) # N > 1, M > 1, M != N


using Base: OneTo

# Helper function to compute the total length of a tuple of indices/axes
function rdims_trailing_product(inds::Tuple)
    total_length = 1
    for ax in inds
        total_length *= length(ax)
    end
    return total_length
end

# Main function equivalent to the recursive rdims logic
function view_dims(::Val{N}, inds::Tuple) where N
    M = length(inds)
    
    # N == 0 case (target is a 0D array/scalar)
    if N == 0
        return ()
    end
    
    # N > M (target dims > source dims): Append OneTo(1) axes
    if N > M
        # Start with all M axes from inds
        new_dims = collect(Any, inds)
        
        # Append (N - M) OneTo(1) axes
        for i in 1:(N - M)
            push!(new_dims, Base.OneTo(1))
        end
        return tuple(new_dims...)
    end
    
    # N == M (target dims == source dims): Simply return the indices
    if N == M
        return inds
    end
    
    # N < M (target dims < source dims): Collapse trailing dimensions
    if N < M
        # Take the first N-1 dimensions as-is
        new_dims = collect(Any, inds[1 : N-1])
        
        # Calculate the product of the remaining trailing dimensions (inds[N] to inds[M])
        trailing_axes = inds[N : M]
        collapsed_length = rdims_trailing_product(trailing_axes)
        
        # The last dimension is a single axis of the collapsed length
        push!(new_dims, Base.OneTo(collapsed_length))
        
        return tuple(new_dims...)
    end
    
    # Should not be reached
    error("Invalid logic in rdims_for_loop")
end

# Example usage (assuming we have 4 input axes, and want 2 output dimensions)
# N=2, M=4
# input_axes = (OneTo(2), OneTo(3), OneTo(4), OneTo(5))
# rdims_for_loop(2, input_axes) 
# -> (OneTo(2), OneTo(3 * 4 * 5))

function reshape(parent::AbstractArray, dims::Dims)
    n = length(parent)
    size_p = size(parent)
    prod(dims) == n || throw( DimensionMismatch( "parent has $n elements, which is incompatible with dimensions $dims ($(prod(dims)) elements)"))
    
    rtn =_reshape((parent, IndexStyle(parent)), dims)
    return rtn
end

function reshape(parent::ReshapedArray42, dims::Dims)
    n = length(parent)
    size_p = size(parent)
    prod(dims) == n || throw( DimensionMismatch( "parent has $n elements, which is incompatible with dimensions $dims ($(prod(dims)) elements)"))
    
    rtn =_reshape((parent.parent, IndexStyle(parent.parent)), dims)
    return rtn
end

# Scalar
function _reshape(p::Tuple{AbstractArray{<:Any,0}, IndexLinear}, dims::Dims)
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


function Base.strides(a::ReshapedArray42)
    # If contingous directly return the strides from the size
    _checkcontiguous(Bool, a) && return _strides_from_size(1, size(a)...)

    # Else we try to merge the dimensin in a 'lazy' way
    parent_size, parent_stride     = size(a.parent), Base.strides(a.parent)
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




# Cette méthode construit la DroppedDimsView
function Base.dropdims(c::AbstractArray; dims::Union{Nothing, Integer, Tuple}=nothing)
    
    # Si dims est non spécifié, on retire toutes les dimensions unitaires.
    if dims === nothing
        dims_to_drop = Tuple(i for (i, d) in enumerate(size(c)) if d == 1)
    
    # Si dims est un scalaire ou un tuple de dimensions fournies par l'utilisateur.
    else
        # S'assurer que les dimensions demandées sont bien de taille 1 dans le AbstractNDArray
        dims_tuple = isa(dims, Integer) ? (dims,) : dims
        for d in dims_tuple
            if size(c, d) != 1
                throw(DimensionMismatch("Dimension $d demandée pour dropdims mais sa taille est $(size(c, d)), pas 1."))
            end
        end
        dims_to_drop = Tuple(dims_tuple)
    end
    
    # Si rien n'est à retirer, retourner la vue originale
    if isempty(dims_to_drop)
        return c
    end

    # Créer et retourner la nouvelle vue paresseuse
    #N_in = ndims(c)
    #N_out = N_in - length(dims_to_drop)
    parent_dims = size(c)
    new_dims = filter(d -> d != 1, parent_dims)
    return reshape(c, new_dims)
end

function offset(a::ReshapedArray42{T, N}, indices::Union{Base.AbstractCartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing; default_offset::Int = 0) where {T, N}
    return _offset(strides(a), indices; default_offset = default_offset)
end

function fromLinearToCartesian(L::Int, a::ReshapedArray42{T, N}) where {T, N}
    L_0 = L - 1

    indices = []

    for d in 1:N
        s_d = size(a, d)
        i_d = Base.mod(L_0, s_d) 
        push!(indices, i_d + 1)
        L_0 = div(L_0, s_d) # division entiere
    end
    I =  CartesianIndex(indices...)
    return I
end

function Base.copy!(dest::AbstractNDArray, src::ReshapedArray42LF)
    @boundscheck size(dest) == size(src)
    for idx in 1:length(src)
        dest[idx] = src[idx]
    end
    return dest
end

function Base.copy!(dest::AbstractNDArray, src::ReshapedArray42)
    @boundscheck size(dest) == size(src)
    for idx in CartesianIndices_42(src)
        dest[idx] = src[idx]
    end
    return dest
end


function Base.show(io::IO, A::ReshapedArray42)
    print(io, "ReshapedArray42 of size ", size(A))
end


Base.lastindex(a::ReshapedArray42{T, N}) where {T, N} = last(eachindex(a)).parentindex
Base.firstindex(a::ReshapedArray42{T, N}) where {T, N} = first(eachindex(a)).parentindex

maybe_reshape_parent(A::AbstractArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
maybe_reshape_parent(A::AbstractArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
maybe_reshape_parent(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
maybe_reshape_parent(A::AbstractArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))