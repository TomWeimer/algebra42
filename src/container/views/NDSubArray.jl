# build tuple of Bool of length ndims(parent), describing which parent dims are kept
function index_dims_for_parent(parent::AbstractArray, indices...)
    # index_dimsum gives a tuple of trues for each *kept* dimension at the front;
    # we want a tuple aligned to parent's dims (kept/dropped info per parent dim).
    # One way: use the Base-style mechanism used in SubArray construction:
    # use rm_singleton_indices with a template tuple of length ndims(parent),
    # then call index_dimsum on that result to produce a tuple length == ndims(parent).
    # But simpler: compute index_dimsum and then pad/truncate to parent's ndims.
    dims = Base.index_dimsum(indices...)    # this returns NTuple{K,Bool} for the effective dims
    # In many Base implementations dims already corresponds to parent's dims, but to be safe:
    K = length(dims)
    P = ndims(parent)
    if K == P
        return dims
    elseif K < P
        # pad trailing falses (dropped dims) to reach P
        return (dims..., ntuple(_ -> false, Val(P - K))...)
    else
        # shouldn't happen: more implied dims than parent dims
        throw(ArgumentError("indices $indices imply more dims ($K) than parent has ($P) parent is $parent"))
    end
end


struct NDSubArray{T,N,P,I,L} <: AbstractNDArray{T,N}
    parent::P
    indices::I

    offset1::Int                   # for linear indexing and pointer, only valid when L==true
    stride1::Int                   # used only for linear indexing

    
    function NDSubArray{T,N,P,I,L}(parent, indices, offset1, stride1) where {T,N,P,I,L}
        @inline
        bools = index_dims_for_parent(parent, indices...)
        check_parent_index_match(parent, bools)
        new(parent, indices, offset1, stride1)
    end
end

check_parent_index_match(parent, indices) = check_parent_index_match(parent, index_ndims(indices...))
check_parent_index_match(parent::AbstractArray{T,N}, ::NTuple{N, Bool}) where {T,N} = nothing
check_parent_index_match(parent, ::NTuple{N, Bool}) where {N} =
    throw(ArgumentError("number of indices ($N) must match the parent dimensionality ($(ndims(parent)))"))

function NDSubArray(parent::AbstractArray, indices::Tuple)
    println("Sort views !! \n indices: ", indices, "going to: ", IndexStyle(Base.viewindexing(indices), IndexStyle(parent)));
    @inline
    NDSubArray(IndexStyle(Base.viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end




# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Basic functions:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Base.strides(a::NDSubArray) = a.strides
# Base.stride(a::NDSubArray, k::Integer) = ( @boundscheck checkindex(Bool, a, k); @inbounds a.strides[k])

function Base.axes(a::NDSubArray{T, N}) where {T, N}
    axes = _axes_from_indices(a.indices)
    if length(axes) == 0 && N != 0
        axes = (Base.OneTo(0), )
    end
    return axes
end

function _axes_from_indices(indices)
    collected_axes = []

    for arg in indices
        if arg isa AbstractArray
            for axis_range in axes(arg)
                push!(collected_axes, axis_range)
            end
        end
    end
    return tuple(collected_axes...)
end

function _size_from_indices(indices)
    return map(length, _axes_from_indices(indices))
end

index_length(idx::Integer) = 0 # Retourne 0 pour indiquer qu'il doit être filtré.
index_length(idx::Base.Slice) = length(idx.indices)
index_length(idx::Any) = length(idx)


function Base.view(a::AbstractNDArray{T, N}, inds...) where {T, N}
    println("We want A[$inds] where A is: ")
    println(a)
    println("-------------- Creation View (indices: $inds) ---------------- ")
    # check bounds
  # check bounds
    J = to_indices(a, inds)

    @boundscheck checkbounds(a, J...)
    
    # drop dimension
    J_2 =  drop_singleton_dimension(J, ndims(a))
    
    # resize parent if needed
    reshaped_parent = maybe_reshape_parent(a, Base.index_ndims(J_2...))
    size_before, size_now = size(a), size(reshaped_parent)
    println("J: $J")
    println("J': $J_2")

    # print the content
    println("content: ", ((size_before != size_now) ? " (reshaped from $size_before to $size_now) " : " " ) * string(reshaped_parent) )

    println("elements befoer beginning the view: reshaped_parent: ndimsA: $(ndims(a)) $reshaped_parent, index_ndims: $(Base.index_ndims(J_2...))")
    # create the view
    V =  create_view(reshaped_parent, J_2...)
    println("offset1: ", V.offset1)
    println("firstindex: ", firstindex(V))
    println("lastindex: ", lastindex(V))
        println("axes: ", axes(V))
            println("indexStyle: ", IndexStyle(V))

    return V
end

function Base.view(a::NDSubArray{T,N,P,I,L}, inds...) where {T,N,P,I,L}
    is_trivial_view(a, inds...) && return  NDSubArray{T, N, P, I, L}(a.parent, a.indices, a.offset1, a.stride1)
    println("We want A[$inds] where A is: ")
    println(a)
    println("-------------- Creation View from NDSubArray (indices: $inds) ---------------- ")
  # check bounds
   # check bounds
    J = to_indices(a, inds)

    @boundscheck checkbounds(a, J...)
    
    # drop dimension
    J_2 =  drop_singleton_dimension(J, ndims(a))
    
    # resize parent if needed
    reshaped_parent = maybe_reshape_parent(a, Base.index_ndims(J_2...))
    size_before, size_now = size(a), size(reshaped_parent)
    println("J: $J")
    println("J': $J_2")

    # print the content
    println("content: ", ((size_before != size_now) ? " (reshaped from $size_before to $size_now) " : " " ) * string(reshaped_parent) )

    println("params: reshaped_parent: ndimsA: $(ndims(a)) $reshaped_parent, index_ndims: $(Base.index_ndims(J_2...))")
    # create the view
    V =  create_view(reshaped_parent, J_2...)
    println("offset1: ", V.offset1)
    println("firstindex: ", firstindex(V))
    println("lastindex: ", lastindex(V))
        println("axes: ", axes(V))
            println("indexStyle: ", IndexStyle(V))

    
    return V
end

create_view(A::AbstractArray, I::Vararg{Any, N}) where {N} = NDSubArray(A, I)

create_view(V::NDSubArray, I::Vararg{Union{Any,AbstractArray{<:Base.AbstractCartesianIndex}} , N}) where {N} = NDSubArray(V, I)


# When we take the view of a view, it's often possible to "reindex" the parent
# view's indices such that we can "pop" the parent view and keep just one layer
# of indirection. But we can't always do this because arrays of `CartesianIndex`
# might span multiple parent indices, making the reindex calculation very hard.
# So we use _maybe_reindex to figure out if there are any arrays of
# `CartesianIndex`, and if so, we punt and keep two layers of indirection.
create_view(V::NDSubArray, I::Vararg{Any, N}) where {N} = maybe_reindex(V, I...)



function reindex(parentindices::Tuple, subindices::Tuple)
    new_indices = []
    subindices_pos = 1

    for idx in parentindices

        # if parent idx have scalar simply add them
        if idx isa Real
            push!(new_indices, idx)
        # if parent idx is a slice pass children idx
        elseif idx isa Base.Slice
            push!(new_indices, subindices[subindices_pos])
            subindices_pos+=1
        # if parent idx is an abstract array of dim n we expect to get n indices in the children indices
        else
            N = ndims(idx)
            if (subindices_pos + N - 1 <= length(subindices))
                child_indices = [subindices[i] for i in subindices_pos:(subindices_pos + N - 1)]
                push!(new_indices, idx[child_indices...]) 
                subindices_pos += N
            else
                throw(ArgumentError("cannot re-index NDSubArray with fewer indices ($N) than dimensions"))
            end
        end
    
    end
    return tuple(new_indices...)
end

function maybe_reindex(V::NDSubArray, I::Vararg{Any, N}) where N
    # check for array of cartesian index
    is_reindexable = true

    formatted_idx = []

    for idx in I
        # We consider array of cartesian index whith one element as array of integer 
        if idx isa AbstractArray{<:Base.CartesianIndex{1}}
            continue
        # In case we see array of cartesian index we do not reindex
        elseif idx isa AbstractArray{<:Base.CartesianIndex}
            is_reindexable = false
            break
        end
    end

    return is_reindexable ? NDSubArray(V.parent, reindex(V.indices, I)) : NDSubArray(V, I)
end

## Re-indexing is the heart of a view, transforming A[i, j][x, y] to A[i[x], j[y]]
#
# Recursively look through the heads of the parent- and sub-indices, considering
# the following cases:
# * Parent index is array  -> re-index that with one or more sub-indices (one per dimension)
# * Parent index is Colon  -> just use the sub-index as provided
# * Parent index is scalar -> that dimension was dropped, so skip the sub-index and use the index as is


function is_trivial_view(S::NDSubArray, inds...)

    # A view is trivially indexed if all indices are simple Slice or Int
    # A StepRange or Array of indices makes it non-trivial
    for (dim, index) in enumerate(inds)
        if !(is_full_slice(S, index, dim))
            return false
        end
    end
    return true
end

function is_full_slice(A::NDSubArray, index::Colon, dim::Int)
    return true;
end

function is_full_slice(A::NDSubArray, index::AbstractRange, dim::Int)
    # 1. Get the official axis for the dimension
    # This is equivalent to what the colon operator (:) would produce.
    default_axis = axes(A, dim)
    
    # 2. Check if the provided range matches the default axis exactly.
    # This works because axes returns a Base.OneTo which is a 1:N range.
    if default_axis == index
        return true
    end
    
    # 3. Handle cases where the index is NOT a simple 1:N range but still covers the full range
    # e.g., if index = 1:1:5, and default_axis = 1:5.
    
    # Check if the start, stop, and step are the default values (1, size, 1)
    if first(default_axis) == first(index) && last(default_axis) == last(index)
        # Check if the step is the default unit step (1)
        if index isa UnitRange || (index isa StepRange && step(index) == 1)
            return true
        end
    end
    
    return false
end

function is_full_slice(A::NDSubArray, index::Any, dim::Int)
    return false;
end


Base.pointer(array::NDSubArray) = pointer(parent(array))
Base.parent(array::NDSubArray) = array.parent
Base.IndexStyle(a::NDSubArray) = IndexStyle(Base.viewindexing(a.indices), Base.IndexStyle(parent(a)))
Base.strides(V::NDSubArray) = substrides(strides(V.parent), V.indices)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get and set element:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
const NDScalarView{T, P, I, L} = NDSubArray{T, 0, P, I, L}

@propagate_inbounds _getElement(a::NDScalarView{T}) where {T} = (
    parent = Base.parent(a); 
    parent[_getIndex(a)...]
)

@propagate_inbounds _setElement!(a::NDScalarView{T}, val::T) where {T} = (
    parent = Base.parent(a);
    parent[_getIndex(a)...] = val
)

# Linear index
@propagate_inbounds _getElement(a::NDSubArray{T, N}, i::Int) where {T, N} = (
    parent = Base.parent(a);
    println("index received: ", i);
    parent[_getIndex(a, i)...]
)

@propagate_inbounds _setElement!(a::NDSubArray{T, N}, val::T, i::Int) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, i)...] = val
)


# Cartesian index
@propagate_inbounds _getElement(a::NDSubArray{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    println("index received: ", I, "index returned: ", _getIndex(a, I));
    parent[_getIndex(a, I)...]
)

@propagate_inbounds _setElement!(a::NDSubArray{T, N}, val::T, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    println("index received: ", I, "index returned: ", _getIndex(a, I));
    parent[_getIndex(a, I)...] = val
)

# 'Normal' index
@propagate_inbounds _getElement(a::NDSubArray{T, N}, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, indices)...]
)

@propagate_inbounds _setElement!(a::NDSubArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, indices)...] = val
)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Linear Index:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

const NDSubArrayFast{T, N, P, I} = NDSubArray{T, N, P, I, true}

Base.IndexStyle(::Type{<:NDSubArrayFast}) = IndexLinear()

function NDSubArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @inline
    # Compute the stride and offset
    #
    #shape = _size_from_indices(indices)
    println("IndexStyle: IndexLinear, N: $N")

    stride = compute_stride1(parent, indices) # (1,) si le parent est (1, 4) est qu'on devient 1D

    parent_strides = strides(parent)

    println("stride1: $stride parent strides: $parent_strides")

    offset = compute_offset1(parent, stride, indices)

    println("previous offset", compute_offset(parent_strides, indices))

    #println("offset: $offset")

    println("axes parent: ", axes(parent))
    # drop dead dimension

    NDSubArrayFast{eltype(P), N, P, I}(parent, indices, offset, stride)
end

# Scalar
@propagate_inbounds _getIndex(a::NDSubArrayFast{T, 0}) where T = a.offset1 + a.stride1 * firstindex(a)

# Linear index
@propagate_inbounds _getIndex(a::NDSubArrayFast{T, N}, i::Int) where {T, N} =  (println( "offset1: ", a.offset1, " i: ", i , "stride1: ", a.stride1, " a.offset1 + a.stride1 * i = ", a.offset1 + a.stride1 * i); a.offset1 + a.stride1 * i)

# @propagate_inbounds _getIndex(a::NDSubArrayFast{T, N}, i::Int) where {T, N} =  parentindex(size(a), strides(a), i, a.offset1)

# Cartesian index
@propagate_inbounds _getIndex(a::NDSubArrayFast{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = _getIndex(a, to_linear_index(axes(a), Tuple(I)))

# 'Normal' index
@propagate_inbounds _getIndex(a::NDSubArrayFast{T, N}, indices::NTuple{N, Any}) where {T, N} = _getIndex(a, to_linear_index(axes(a), indices))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Cartesian Index:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

const NDSubArraySlow{T, N, P, I} = NDSubArray{T, N, P, I, false}

Base.IndexStyle(::Type{<:NDSubArraySlow}) = CartesianIndex()

# If the view accept cartesian index
function NDSubArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @inline
    println("IndexStyle: IndexCartesian, N: $N")

    parent_strides = strides(Base.parent(parent))

    println("stride1: 0 parent strides: $parent_strides")

    #offset = compute_offset(parent_strides, indices)

   # println("offset: $offset")

    println("axes: ", axes(parent))

   
    NDSubArray{eltype(P), N, P, I, false}(parent, indices, 0, 0)
end

# scalar
@propagate_inbounds _getIndex(a::NDSubArraySlow{T, 0}) where T = reindex(a.indices, firstindex(a))

# Linear index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T, N}, i::Int) where {T, N} =  reindex(a.indices, tuple(_linear_to_cartesian(size(a), i)...))

# Cartesian index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = (I = reindex(a.indices, Tuple(I)); println("returned: ", I, " linear: ", to_linear_index(axes(a), Tuple(I))); I )

# 'Normal' index
@propagate_inbounds _getIndex(a::NDSubArraySlow{T, N}, indices::NTuple{N, Int}) where {T, N} = reindex(a.indices, indices)


function offset(a::NDSubArray{T, N}, indices::Union{Base.AbstractCartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing; default_offset::Int = 0) where {T, N}
    return _offset(strides(a), indices; default_offset = a.offset1)
end

# using Test

# # --- Setup: Parent 3D Array ---
# # Create a 3x3x3 array (27 elements) for testing.
# # Elements are 1 to 27 in column-major order.
# A = reshape(collect(1:27), 3, 3, 3)

# # Base values in A for reference:
# # A[:, :, 1] = [ 1 4 7; 2 5 8; 3 6 9 ]
# # A[:, :, 2] = [ 10 13 16; 11 14 17; 12 15 18 ]
# # A[:, :, 3] = [ 19 22 25; 20 23 26; 21 24 27 ]

# Test.@testset "Array View Features" begin

#     # --- 1. Regular/Strided View Test (Constant Multiplicative Increment) ---
#     # V_reg is created using only ranges (2:3, 1:2) and a fixed index (3).
#     # This view is guaranteed to be regularly strided (MI is possible).
#     V_reg = view(A, 2:3, 1:2, 3)

#     Test.@testset "Regular View (V_reg)" begin
#         Test.@test size(V_reg) == (2, 2)
#         Test.@test ndims(V_reg) == 2 # Fixed index 3 is dropped

#         # Check element access (V_reg[i, j] -> A[i+1, j, 3])
#         # V_reg[1, 1] should be A[2, 1, 3] = 20
#         Test.@test V_reg[1, 1] == 20
#         # V_reg[2, 2] should be A[3, 2, 3] = 24
#         Test.@test V_reg[2, 2] == 24
#     end

#     # --- 2. Fancy Indexing View Test (Indirection Required) ---
#     # V_fancy uses arbitrary vectors ([3, 1], [3, 1]).
#     # This view is *not* regularly strided (NO constant MI).
#     V_fancy = view(A, [3, 1], [3, 1], 2)
#      Idx1 = [3, 1]
#      Idx2 = [3, 1]

#     Test.Test.@testset "Fancy View (V_fancy)" begin
#         Test.@test size(V_fancy) == (2, 2)

#         # Check element access (requires lookup/indirection)
#         # V_fancy[1, 1] -> A[Idx1[1], Idx2[1], 2] -> A[3, 3, 2] = 18
#         Test.@test V_fancy[1, 1] == 18
#         # V_fancy[2, 1] -> A[Idx1[2], Idx2[1], 2] -> A[1, 3, 2] = 16
#         Test.@test V_fancy[2, 1] == 16
#         # V_fancy[2, 2] -> A[Idx1[2], Idx2[2], 2] -> A[1, 1, 2] = 10
#         Test.@test V_fancy[2, 2] == 10
#     end

#     # --- 3. Mutation and Sharing Memory Test ---

#     Test.@testset "Mutation Check" begin
#         # 3.1. Mutate the Fancy View
#         # We change V_fancy[1, 1], which corresponds to A[3, 3, 2] (original value: 18)
#         V_fancy[1, 1] = 999
#         Test.@test V_fancy[1, 1] == 999
#         Test.@test A[3, 3, 2] == 999  # CRITICAL: Parent array is updated

#         # 3.2. Mutate the Regular View
#         # We change V_reg[1, 1], which corresponds to A[2, 1, 3] (original value: 20)
#         V_reg[1, 1] = 888
#         Test.@test V_reg[1, 1] == 888
#         Test.@test A[2, 1, 3] == 888  # CRITICAL: Parent array is updated

#         # 3.3. Mutate the Parent and check the Views
#         A[1, 1, 2] = 777
#         # A[1, 1, 2] is V_fancy[2, 2]
#         Test.@test V_fancy[2, 2] == 777
#     end
# end



# abstract type AbstractNDView{T, N, ParentType <: AbstractArray{T}, IndicesType, LinearIndex} <: AbstractNDArray{T, N} end
# const AbstractScalarView{T, P, I, L} = AbstractNDView{T, 0, P, I, L}
# const AbstractScalarViewFast{T, N, P, I} = AbstractNDView{T, 0, P, I, true}
# const AbstractScalarViewSlow{T, N, P, I} = AbstractNDView{T, 0, P, I, false}


# const NDFastView{T, N, P, I} = AbstractNDView{T, N, P, I, true} 
# const NDSlowView{T, N, P, I} = AbstractNDView{T, N, P, I, false} 


function to_linear_index(axes::NTuple{N,AbstractUnitRange}, I::NTuple{N,Int}) where {N}
    stride = 1
    lin = 1
    for d in 1:N
        ax = axes[d]
        i = I[d]
        lin += (i - first(ax)) * stride
        stride *= length(ax)
    end
    return lin
end

function substrides(parent_strides::NTuple{N, Int}, indices::NTuple{M, Any}) where {N, M}
    s = 1

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