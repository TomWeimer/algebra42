
abstract type AbstractNDArrayView{T, N, M} <: AbstractNDArray{T, N} end


"""
    NDArrayView{DType, N, M} <: AbstractNDArray{DType, N}

A **non-owning view** into an existing `NDArray` or another `NDArrayView`.  
It provides a *window* onto the parent array's memory without copying data.

`NDArrayView` allows efficient slicing, subsetting, and reshaping operations
that reference the same underlying buffer as the parent array.

# Type Parameters
- `DType` — The element type of the array (e.g., `Float64`, `Int32`).
- `N` — The number of dimensions of the view (may differ from the parent).
- `M` — The number of dimensions of the parent array.

# Fields
- `parent::AbstractNDArray{DType, M}`  
  The parent array or view that owns the actual data buffer.

- `shape::NTuple{N}`  
  The shape (size per dimension) of the view.

- `initial_offset::Int`  
  The linear offset into the parent's memory where the view starts.
  Computed based on the first index of each dimension.

- `strides::Tuple{Vararg{Int}}`  
  The stride (step size in memory) for each view dimension, inherited
  and possibly scaled from the parent's strides.

# Notes
- The view does **not** own its memory — modifying it will affect the parent array.
- If the view covers the same region as its parent and has the same strides and offset,
  it is considered a *trivial view* (i.e., equivalent to the parent).
- The rank `N` may differ from `M` in cases where integer indexing drops dimensions.

# Complexity
- **Construction:** `O(N)` (to compute shape, strides, and offset)
- **Memory:** `O(1)` (no data copy; only metadata stored)

# Example
```julia
A = NDArray(rand(4, 4))
V = NDArrayView(A, (2:3, 1:2), initial_offset=5, strides=(4, 1))
V[1, 1] == A[2, 1]  # true — same memory, different view
"""
mutable struct NDArrayView{DType, N, M} <: AbstractNDArrayView{DType, N, M}
   parent::AbstractNDArray{DType, M}
   shape::NTuple{N}
   initial_offset::Int              # starting position in the parent’s linear buffer
   strides::Tuple{Vararg{Int}}
end

Base.size(array::NDArrayView) = array.shape

Base.ndims(array::NDArrayView{T, N}) where {T, N} = N

Base.eltype(A::NDArrayView{T}) where {T} = T

Base.pointer(A::NDArrayView{T}) where T = pointer(A.parent)

function Base.stride(A::NDArrayView{T, N}, k::Integer) where {T,N}
   (1 <= k <= N) || throw(ArgumentError("The index k is out of bounds"))
   return A.strides[k]
end


function Base.strides(A::NDArrayView{T, N}) where {T,N}
   return A.strides
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructors:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function create_view_from_indices(parent::AbstractNDArray{T, M}, indices::Vararg{Colon, M}) where {T, M}
    return NDArrayView{T, M, M}(parent, size(parent), 0, strides(parent))
end


create_view_from_indices(parent::NDArrayView{T, M}, indices::Vararg{Any, M}) where {T, M} = create_view_from_indices(parent, indices...; default_init_offset = parent.initial_offset)

function create_view_from_indices(parent::AbstractNDArray{T, M}, indices::Vararg{Any, M}; default_init_offset = 0) where {T, M}

    # 1. Input Validation and Preparation
    # Convert Vararg to a tuple for consistent handling
    index_tuple = indices 

    # 2. Step 1: Calculate the Absolute Initial Offset
    # This must be done first, as it uses ALL indices (including the ones that drop dimensions)
    # and the parent's original strides to find the view's starting byte.
    initial_offset = obtain_initial_offset(parent, index_tuple) + default_init_offset

    # 3. Step 2: Create the Final View Object
    # This function uses the same index_tuple and the newly calculated offset.
    # It handles the complex logic of:
    #   - Dropping dimensions (when index is an Integer).
    #   - Calculating the new strides (when index is a Range/Slice).
    #   - Determining the final shape and rank (N_new).
    return createView(parent, index_tuple, initial_offset)
end

function createView(parent::AbstractNDArray{T, M},  indices::NTuple{M, Any}, initial_offset::Int) where {T, M}
    new_shape_list = []
    new_strides_list = []
    
    for k in 1:M
        idx = indices[k]
        original_stride = parent.strides[k]

        # Dimension Dropping: Do nothing, don't include in shape/strides
        if idx isa Integer 
            continue
        # Dimension Preserving (AbstractRange or Colon)
        else
            # Determine the step and length for the stride update and new shape
            step_k = idx isa AbstractRange ? step(idx) : 1
            
            # 1. Stride: compute the strides of the view
            new_stride_k = original_stride * abs(step_k)
            push!(new_strides_list, new_stride_k)

            # 2. Shape: Calculate the shape of the view
            len_k = idx isa Colon ? size(parent, k) : length(idx)  
            push!(new_shape_list, len_k)
        end
    end
    
    # Convert lists to tuples and determine the new rank N_new
    new_shape, new_strides = tuple(new_shape_list...), tuple(new_strides_list...)

    # The view needs the parent reference to access data
    return NDArrayView{T, length(new_shape), M}(parent, new_shape, initial_offset, new_strides)
end

# A function to create a new view, given ALL necessary view metadata.
function createViewFromMetadata(parent::AbstractNDArray{T, M}, new_shape::NTuple{N, Int}, initial_offset::Int, new_strides::NTuple{N, Int}) where {T, N, M}
    
    # 1. Validation (Optional but Recommended)
    if length(new_shape) != length(new_strides)
        error("Shape and strides must have the same length (N).")
    end
    
    # 2. Construction
    # The 'N' is the rank of the view, and 'M' is the rank of the parent.
    return NDArrayView{T, N, M}(
        parent,          # The parent array (data source)
        new_shape,       # The shape of the view (N-tuple)
        initial_offset,  # The absolute memory offset to the view's (1, 1, ...) element
        new_strides      # The memory stride in bytes for each dimension of the view (N-tuple)
    )
end

function create_view_from_shape(parent::AbstractNDArray{T, M}, shape::NTuple{N}, initial_offset = 0) where {T, M, N} 
    return createViewFromMetadata(parent, shape,  initial_offset, _compute_strides(shape))
end


create_view_from_shape(parent::NDArrayView{T, M}, shape::NTuple{N}) where {T, M, N} = create_view_from_shape(parent, shape, parent.initial_offset)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Set Index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# From a view that is suppose to be a scalar obtain the right element
function Base.setindex!(v::NDArrayView{DType,0}, val::DType, ::Tuple{}) where {DType} 
    _setElement!(v.parent, _offset(v, ()), val)
end

# From a view change the index with the offset
Base.setindex!(v::NDArrayView{DType}, val::DType, indices::Vararg{Int}) where {DType} = _setElement!(v.parent, _offset(v,  indices...), val)


Base.setindex!(v::NDArrayView{DType,N}, val::DType, I::CartesianIndex) where {DType,N} = _setElement!(v.parent, _offset(v, I), val)


_setElement!(a::NDArrayView{DType,N}, index::Int, val) where {DType,N} = _setElement!(a.parent, a.initial_offset + index, val)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get Index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


function Base.getindex(v::NDArrayView{DType,0}, ::Tuple{}) where {DType}
    return item(v.parent, _offset(v, ()))
end

# Accessing a 0-dimensional view with a single integer is invalid
function Base.getindex(::NDArrayView{DType,0}, indices::Int) where {DType}  # return an error
    throw(DomainError(
        indices,
        "Cannot index a 0-dimensional view with a single integer. " * "Use `view[()]` or `view.item()` to access the scalar value."
    ))
end

# Accessing a non 0-Dimensional view with an empty tuple
function Base.getindex(::NDArrayView{DType,N}, args::Tuple{}) where {DType,N}
    throw(DomainError(
        args,
        "Empty tuple indexing is only valid for 0-dimensional arrays. "
    ))
end 

# Accessing a 0-dimensional view with a colon or ranges
Base.getindex(::NDArrayView{DType,0}, ::Vararg{ViewIndices}) where {DType} = throw(DomainError(
    ":",
    "Cannot use `:` or ranges to index a 0-dimensional view. " *
    "Use `view[()]` or `view.item()` to access the scalar value."
))

# Normal indexing
function Base.getindex(v::NDArrayView{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element
    # Check if the index are valid
    (elementsAreValid(size(v), indices...)) || throw(DomainError("Indexes out of bounds"))

   # println("v.parent: ", v.parent, " indices: ", indices, " offset: ", _offset(v, indices...))

    
    # If the index is valid return the associated elements
    return item(v.parent, _offset(v, indices...))
end

# Must return an element
function Base.getindex(v::NDArrayView{T, N}, I::CartesianIndex{N}) where {T, N}
    # If the index is valid return the associated elements
    return item(v.parent, _offset(v, I))
end

# Must expand indexing
function Base.getindex(v::NDArrayView{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
    # Check if the number of index is valid
    dimIndices, dimView = length(indices), ndims(v)

    ( dimIndices <=  dimView) || throw(DomainError("The number of index provided is too big"))

    # Expand with colon if needed
    indicesStretched = (dimIndices == dimView) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimView)

    return v[indicesStretched...]
end

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



# Create a view from shape
function reshape(a::AbstractNDArray{T, M}, shape::NTuple{N})  where {T, M, N} 
    prod(size(a)) ==  prod(shape) || throw(ArgumentError("Can't resize the array with the shape given"))
    return create_view_from_shape(parent, shape)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       _offset:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #



function _offset(view::NDArrayView{T, 0}, ::Tuple{})::Int where{T}
    return view.initial_offset + 1# start at initial offset
end


function _offset(view::NDArrayView{T, N}, indices::Vararg{Int, N})::Int where{T, N}
    offset = view.initial_offset  # base offset in parent array
    for i in 1:ndims(view)
      #  println("enter in the loop: ", offset)
        offset += (indices[i] - 1) * view.strides[i]
    end
    return offset + 1
end

# Obtain the index ( flat index ) from cartesian index
_offset(view::NDArrayView{T, N}, I::CartesianIndex{N}) where {T, N} = _offset(view, Tuple(I)...)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Initial Offset:                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #



# Calculates the initial offset using the parent's stride and the starting index of every dimension.
obtain_initial_offset(p::AbstractNDArray{T, M}, inds::NTuple{M, Any}) where {T, M} = sum((obtain_sk(idx) - 1) * p.strides[k] for (k, idx) in enumerate(inds))


# Return the starting index for various types
obtain_sk(idx::Integer)      = idx
obtain_sk(r::AbstractRange)  = first(r)
obtain_sk(::Colon)           = 1
obtain_sk(::Any)             = error("Unsupported index type")


"""
    item(a::NDArray, idx...)

Accesses an element of the NDArray using the provided indices.
For scalars, only empty indices are allowed.
"""
item(v::NDArrayView{DType}, idx::Vararg{Int,1}) where {DType} =  item(v.parent, idx[1]) 