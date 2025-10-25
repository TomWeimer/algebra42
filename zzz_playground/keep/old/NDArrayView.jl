
include("AbstractView.jl")

abstract type AbstractCartesianIndex{N} end # This is a hacky forward declaration for CartesianIndex
const ViewIndex = Union{Real, AbstractArray}
const ScalarIndex = Real


struct SubNDArray{T,N, P <: AbstractArray, I,L} <: AbstractNDView{T,N, P, I, L}
    parent::P
    indices::I

    offset1::Int       # for linear indexing and pointer, only valid when L==true
    stride1::Int       # used only for linear indexing

    
    function SubNDArray{T,N,P,I,L}(parent, indices, offset1, stride1) where {T,N,P,I,L}
        @inline
        check_parent_index_match(parent, indices)
        new(parent, indices, offset1, stride1)
    end
end



# Check the number of dimension between parent and indices
check_parent_index_match(parent::AbstractArray{T, N}, ::NTuple{N, Any}) where {T, N} = nothing

check_parent_index_match(parent, ::NTuple{N, Any}) where N = (
    throw(ArgumentError("The number of indices do not match the dimensions of the array $parent"))
)

# Function creating the view
function view(A::AbstractArray, I::Vararg{Any,M}) where {M}
    @boundscheck checkbounds(A, I...)
    @inbounds indices = to_indices(A, I)
    return SubNDArray(A, indices)
end


# Constructor
function SubNDArray(parent::AbstractArray, indices::Tuple)
    @inline
    SubNDArray(IndexStyle(Base.viewindexing(indices), Base.IndexStyle(parent)), parent, Base.ensure_indexable(indices), Base.index_dimsum(indices...))
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       SubNDArrayFast:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

const SubNDArrayFast{T,N,P,I} = SubNDArray{T, N, P, I, true}

# If the view accept linear index
function SubNDArray(::IndexLinear, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @inline
    # Compute the stride and offset
    stride = compute_stride(parent, indices)
    offset = compute_offset(strides(parent), indices)
    SubNDArrayFast{eltype(P), N, P, I}(parent, indices, offset, stride)
end

function compute_stride(parent::AbstractArray, indices::NTuple{N}) where N
    new_strides_list = []

    for k in 1:N
        idx = indices[k]

        # Dimension Dropping: Do nothing, don't include in shape/strides
        if idx isa Integer 
            continue
        end
        # Compute strides and shape from the indices
        step_k = idx isa AbstractRange ?  abs(step(idx)) : 1
        new_stride_k = strides[k] * step_k
        push!(new_strides_list, new_stride_k)
    end
    
    # Convert lists to tuples and determine the new rank N_new
    return tuple(new_strides_list...)
end

function compute_offset(strides::AbstractArray{Int, N}, inds::NTuple{N}) where {N} 
    return sum((starting_idx(idx) - 1) * strides[k] for (k, idx) in enumerate(inds))
end

# Return the starting index for various types
starting_idx(idx::Integer)      = idx
starting_idx(r::AbstractRange)  = first(r)
starting_idx(::Base.Slice)      = 1
starting_idx(::Colon)           = 1
starting_idx(::Any)             = error("Unsupported index type")


Base.IndexStyle(a::SubNDArrayFast) = IndexLinear()

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       SubNDArraySlow:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

const SubNDArraySlow{T,N,P,I} = SubNDArray{T, N, P, I, false}

# If the view accept cartesian index
function SubNDArray(::IndexCartesian, parent::P, indices::I, ::NTuple{N,Any}) where {P,I,N}
    @inline
    SubNDArray{eltype(P), N, P, I, false}(parent, indices, 0, 0)
end

Base.IndexStyle(a::SubNDArraySlow) = CartesianIndex()


function Base.size(a::AbstractNDView)
    dims_tuple = ()
    for idx in a.indices
        len = index_length(idx)
        if len > 0
            dims_tuple = (dims_tuple..., len)
        end
    end
    return dims_tuple
end


index_length(idx::Integer) = 0 # Retourne 0 pour indiquer qu'il doit être filtré.
index_length(idx::Base.Slice) = length(idx.indices)
index_length(idx::Any) = length(idx)


Base.parent(array::AbstractNDView) = array.parent
Base.IndexStyle(a::AbstractNDView) = IndexStyle(parent(a))
Base.strides(a::AbstractNDView) = a.strides


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
































# """
#     item(a::NDArray, idx...)

# Accesses an element of the NDArray using the provided indices.
# For scalars, only empty indices are allowed.
# """
# item(v::NDArrayView{DType}, idx::Vararg{Int,1}) where {DType} =  item(v.parent, idx[1]) 



# A function to create a new view, given ALL necessary view metadata.
# function createViewFromMetadata(parent::AbstractNDArray{T, M}, new_shape::NTuple{N, Int}, initial_offset::Int, new_strides::NTuple{N, Int}) where {T, N, M}
    
#     # 1. Validation (Optional but Recommended)
#     if length(new_shape) != length(new_strides)
#         error("Shape and strides must have the same length (N).")
#     end
    
#     # 2. Construction
#     # The 'N' is the rank of the view, and 'M' is the rank of the parent.
#     return NDArrayView{T, N, M}(
#         parent,          # The parent array (data source)
#         new_shape,       # The shape of the view (N-tuple)
#         initial_offset,  # The absolute memory offset to the view's (1, 1, ...) element
#         new_strides      # The memory stride in bytes for each dimension of the view (N-tuple)
#     )
# end

# function create_view_from_shape(parent::AbstractNDArray{T, M}, shape::NTuple{N}, initial_offset = 0) where {T, M, N} 
#     return createViewFromMetadata(parent, shape,  initial_offset, _compute_strides(shape))
# end


# create_view_from_shape(parent::NDArrayView{T, M}, shape::NTuple{N}) where {T, M, N} = create_view_from_shape(parent, shape, parent.initial_offset)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Set Index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# # From a view that is suppose to be a scalar obtain the right element
# function Base.setindex!(v::NDArrayView{DType,0}, val::DType, ::Tuple{}) where {DType} 
#     _setElement!(v.parent, _offset(v, ()), val)
# end

# # From a view change the index with the offset
# Base.setindex!(v::NDArrayView{DType}, val::DType, indices::Vararg{Int}) where {DType} = _setElement!(v.parent, _offset(v,  indices...), val)


# Base.setindex!(v::NDArrayView{DType,N}, val::DType, I::CartesianIndex) where {DType,N} = _setElement!(v.parent, _offset(v, I), val)



# # Accessing a 0-dimensional view with a single integer is invalid
# function Base.getindex(::NDArrayView{DType,0}, indices::Int) where {DType}  # return an error
#     throw(DomainError(
#         indices,
#         "Cannot index a 0-dimensional view with a single integer. " * "Use `view[()]` or `view.item()` to access the scalar value."
#     ))
# end

# # Accessing a non 0-Dimensional view with an empty tuple
# function Base.getindex(::NDArrayView{DType,N}, args::Tuple{}) where {DType,N}
#     throw(DomainError(
#         args,
#         "Empty tuple indexing is only valid for 0-dimensional arrays. "
#     ))
# end 

# # Accessing a 0-dimensional view with a colon or ranges
# Base.getindex(::NDArrayView{DType,0}, ::Vararg{ViewIndices}) where {DType} = throw(DomainError(
#     ":",
#     "Cannot use `:` or ranges to index a 0-dimensional view. " *
#     "Use `view[()]` or `view.item()` to access the scalar value."
# ))

# Normal indexing
# function Base.getindex(v::NDArrayView{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element
#     # Check if the index are valid
#     (elementsAreValid(size(v), indices...)) || throw(DomainError("Indexes out of bounds"))

#    # println("v.parent: ", v.parent, " indices: ", indices, " offset: ", _offset(v, indices...))

    
#     # If the index is valid return the associated elements
#     return item(v.parent, _offset(v, indices...))
# end

# Must return an element
# function Base.getindex(v::NDArrayView{T, N}, I::CartesianIndex{N}) where {T, N}
#     # If the index is valid return the associated elements
#     return item(v.parent, _offset(v, I))
# end

# # Must expand indexing
# function Base.getindex(v::NDArrayView{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
#     # Check if the number of index is valid
#     dimIndices, dimView = length(indices), ndims(v)

#     ( dimIndices <=  dimView) || throw(DomainError("The number of index provided is too big"))

#     # Expand with colon if needed
#     indicesStretched = (dimIndices == dimView) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimView)

#     return v[indicesStretched...]
# end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Create View:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# function Base.view(parent::AbstractNDArray, inds::Vararg{ViewIndices})
#     view = create_view_from_indices(parent, inds...)

#     return view
# end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Reshape:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #



# # Create a view from shape
# function reshape(a::AbstractNDArray{T, M}, shape::NTuple{N})  where {T, M, N} 
#     prod(size(a)) ==  prod(shape) || throw(ArgumentError("Can't resize the array with the shape given"))
#     return create_view_from_shape(parent, shape)
# end


# _setElement!(a::NDArrayView{DType,N}, index::Int, val) where {DType,N} = _setElement!(a.parent, a.initial_offset + index, val)
