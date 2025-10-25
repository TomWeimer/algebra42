

# Index types:
# -----------

# Return a single elements of the array ( if enough are provided )
const ElementIndex = Int                          

# Return a view if:
# - a colon is present
# - the number of indices given is less than the number of array's dimensions
const ViewIndices = Union{AbstractRange, Int, Colon} 

# Fancy indices are arrays of integer or boolean
const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# Return a copy of the array, if at least one fancy index is present
const CopyIndices = Union{AbstractRange, Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}



# Accessing 0-dimensional array:
# -----------------------------

"""
    Base.getindex(array::NDArray{DType,0}, ::Tuple{}) where {DType}

Returns the single element from a 0-dimensional `NDArray`.

A 0-dimensional ndarray represents a single, scalar value. According to Julia's
indexing rules, the correct way to retrieve this element is by using `array[]`.
This method enable the numpy's syntax `array[()]` if the array is 0-dimensional.
"""
function Base.getindex(array::NDArray{DType,0}, ::Tuple{}) where {DType}
    return item(array)
end

"""
    Base.getindex(::NDArray{DType,0}, indices::Int) where {DType}

Throw an error if we try to access a 0-dimensional ndarrray with the syntax `array[1]`

"""
function Base.getindex(::NDArray{DType,0}, indices::Int) where {DType}  # return an error
    throw(DomainError("accessing by index with a scalar is only possible as ndarray[()] or ndarray.item()"))
end


"""
    Base.getindex(::NDArray{DType, N}, args::Tuple{}) where {DType, N}

Prevents the use of a 0-dimensional index `array[()]` on a multi-dimensional `NDArray`.

This method serves as a safety check, explicitly throwing a `DomainError` when a user
attempts to access a non-scalar array with an empty tuple. This syntax is
reserved for 0-dimensional arrays to retrieve their single element.

The function correctly identifies that `N > 0` and provides a clear error message
explaining why this operation is invalid for multi-dimensional arrays.
"""
function Base.getindex(::NDArray{DType,N}, args::Tuple{}) where {DType,N} # return an error
    # Check that the tuple is not empty
    isempty(args) && throw(DomainError("The dimension is not 0 ndarray[()] works only on scalar"))
    # Throw an error the element can not be accessed with a tuple
    throw(DomainError("The element cannnot be accessed with tuple"))
end


"""
    Base.getindex(::NDArray{DType,0}, ::Vararg{ViewIndices}) where {DType, N}

Prevents the use of slices in 0-dimensional `NDArray` such as `array[:]`.

This method serves as a safety check, explicitly throwing a `DomainError` when a user
attempts to slice the 0-dimensional array.
"""
Base.getindex(::NDArray{DType,0}, ::Vararg{ViewIndices}) where {DType} = throw(DomainError("Cannot access to scalar with ndarray[:]"))



# Accessing N-dimensional array:
# -----------------------------

"""
    Base.getindex(array::NDArray{DType, N}, indices::Vararg{ElementIndex, N}) where {DType, N}

Returns a single element from an N-dimensional `NDArray` using a fixed number of integer indices.

This method overloads the `getindex` function to enable direct element access using a
tuple of indices, such as `my_array[1, 2, 3]`. It is a fundamental part of the
NDArray's API, providing the standard Julia-like indexing behavior.

# Arguments
- `array::NDArray{DType, N}`: The N-dimensional array from which to retrieve an element.
- `indices::Vararg{ElementIndex, N}`: A variadic tuple of integer indices. The
  number of indices provided must match the dimension `N` of the array.

# Returns
- A single element of type `DType` at the specified index.

# Throws
- `DomainError`: Thrown if the provided indices are out of bounds or if the
  number of indices does not match the array's dimension `N`.
"""
function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element

    # Check that the indices are not empty
    isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

    # Check if the index are valid
    (elementsAreValid(size(array), indices...)) || throw(DomainError("Indexes out of bounds"))

    # If the index is valid return the associated elements
    return array.content[_offset(array, indices...)]
end


"""
    Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}

Returns a view or a slice of an `NDArray` using a partial set of integer indices.

This method overloads `getindex` to provide flexible array slicing, similar to Python's NumPy.
If the number of indices provided is less than the dimension `N` of the array,
the remaining dimensions are implicitly treated as full slices (`:`). This allows for
simple operations like `my_array[1, 2]` to return a sub-array (a view) rather than throwing an error.

The function works by "stretching" the provided indices into a full `N`-element tuple
by appending a `Colon()` object for each missing index. This new tuple is then
used to perform the actual slicing operation on the underlying data. 

# Arguments
- `array::NDArray{DType,N}`: The multi-dimensional array to be sliced.
- `indices::Vararg{ElementIndex}`: A variadic tuple of integer indices. The number
  of indices can be from 1 up to the dimension `N`.

# Returns
- A view on the orignal `NDArray`.

# Throws
- `DomainError`: Thrown if the number of indices provided is greater than the
  array's dimension `N`, or if the indices are empty.
"""
function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
    # Check that the indices are not empty
    isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

    # Check if the number of index is valid
    dimIndices, dimArray = length(indices), ndims(array)

    (dimIndices <= dimArray) || throw(DomainError("The number of index provided is too big"))

    indicesStretched = (dimIndices == dimArray) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimArray)

    # If the index is valid return the associated elements
    return array[indicesStretched...]
end


"""
    Base.getindex(array::NDArray{DType, N}, indices::Vararg{ViewIndices}) where {DType, N}

Performs array slicing and returns a memory-efficient `view` of the `NDArray`.

This method enables advanced indexing using `AbstractRange` (e.g., `1:3`) and `Colon` (`:`).
Instead of creating a full copy of the data, it returns a lazy `view` into the original array. 
This makes slicing operations very fast and memory-efficient, especially for large arrays.

# Arguments
- `array::NDArray{DType, N}`: The N-dimensional array to be sliced.
- `indices::Vararg{ViewIndices}`: A variadic tuple of indices. `ViewIndices`
  can be a mix of `Int`, `Colon`, and `AbstractRange`.

# Returns
- A new `NDArray` object representing a `view` of the original array.

# Throws
- `DomainError`: Thrown if the provided indices are out of bounds.
"""
function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ViewIndices}) where {DType,N}
    # Check that the indices are not empty
    isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
    # Check if the index are valid
    #@infiltrate
    _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
    # If the index is valid return the associated elements
    return @view array[indices...]
end


# special indices:
# ----------------

"""
    Base.getindex(A::NDArray, I::CartesianIndex)

Returns a single element from an `NDArray` using a `CartesianIndex` object.

A `CartesianIndex` is a type that holds a multi-dimensional index. It is commonly
used when iterating over the elements of a multi-dimensional array, as it provides
a convenient way to represent each element's position. This method overloads `getindex`
to allow direct element access using this type.

# Arguments
- `A::NDArray`: The N-dimensional array from which to retrieve the element.
- `I::CartesianIndex`: The `CartesianIndex` object specifying the position of the
  desired element.

# Returns
- A single element from the array at the position specified by `I`.
"""
function Base.getindex(A::NDArray{T, N}, I::CartesianIndex{N}) where {T, N}
    A.content[_offset(A, I)]
end

# fancy indices:
# ----------------

"""
    Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}

Performs "fancy indexing" on an `NDArray` using a single index, such as a boolean array or a vector of integers.

This method overloads `getindex` to provide a powerful and flexible way to select
multiple non-contiguous elements from an array. It delegates the complex logic of
fancy indexing to a separate `fancyIndexing` function, which handles the
conversion and broadcasting of indices.

# Arguments
- `array::NDArray{DType,N}`: The N-dimensional array from which to select elements.
- `indices::CopyIndices`: A single fancy index, which can be an array of integers,
  a boolean array, or any type that represents a collection of indices.

# Returns
- A new `NDArray` containing the selected elements. This is a copy of the data,
  not a view.

# Implementation Details
- The function first converts the input `indices` into a standardized format
  using the `convertIndices` helper function.
- It then calls `fancyIndexing` to perform the actual element selection and
  return the new `NDArray`. 

# Example
```julia
nd_array = NDArray([10, 20, 30, 40, 50])

# Select elements at index 1 and 3
subset_vector = nd_array[[1, 3]] 
# subset_vector will be a new NDArray with content [10, 30]

# Select elements based on a boolean mask
bool_mask = [true, false, true, false, false]
subset_bool = nd_array[bool_mask]
# subset_bool will be a new NDArray with content [10, 30]
"""
function Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}
    return fancy_index(array, indices)
end


"""
    Base.getindex(array::NDArray{DType, N}, indices::Vararg{CopyIndices}) where {DType, N}

Performs advanced indexing or fancy indexing on an `NDArray` using a combination of multiple indices.

This method handles complex indexing scenarios where a user provides multiple
indices, such as a mix of vectors and boolean masks. The function is designed to
be the entry point for all advanced indexing operations.

It works by delegating the heavy lifting to two helper functions: `convertIndices`,
which standardizes the input indices, and `fancyIndexing`, which performs the
actual selection and broadcasting logic to create the final result. 

# Arguments
- `array::NDArray{DType, N}`: The N-dimensional array from which to select elements.
- `indices::Vararg{CopyIndices}`: A variadic tuple of fancy indices. This can be
  a mix of arrays of integers, boolean arrays, or other indexable types.

# Returns
- A new `NDArray` containing the selected elements. This is a copy of the data,
  not a view.
"""
function Base.getindex(array::NDArray{DType,N}, indices::Vararg{CopyIndices}) where {DType,N}
   # println("enter motherfucker2")
    return fancy_index(array, indices)
end


"""
    fancy_index(A::NDArray, indices...)

Advanced/fancy indexing for NDArray using MultiIter.

# Arguments
- `A::NDArray`: The array to index.
- `indices...`: A variable number of indices, which can be
  - Integers (single element),
  - Ranges,
  - Vectors of integers (fancy indexing),
  - Boolean arrays (masking).

# Returns
- `NDArray` with elements selected according to `indices`.

# Notes
- Supports broadcasting across index arrays.
- MultiIter is used internally to handle multiple index arrays.

# Complexity
- Let `n` be the total number of selected elements.
- **Time complexity:** O(n) (linear in the number of elements to retrieve)
- **Space complexity:** O(n) (for the output array)
"""
# function fancy_index(A::NDArray, indices)

#     # Step 2: Compute broadcasted shape of indices
#     output_shape = obtain_broadcast_shape(indices)

#     println("output_shape: $output_shape")

#     # Step 3: Initialize output array
#     out = NDArray{eltype(A)}(output_shape)

#     # Step 4: Create multi-iterator over index arrays
#     mit = MultiIter(indices...)

#     # Step 5: Iterate and extract elements
#     for (out_idx, idx_vals) in zip(CartesianIndices(output_shape), mit)
#         linear_idx = Tuple(idx_vals)
#         out[out_idx] = A[linear_idx...]  # use splatting for multi-dimensional indexing
#     end

#     return out
# end


# Internal methods:
# -----------------

function convertIndices(shape::Tuple, index::CopyIndices)
    return (convertIndex(shape, 1, index),)
end

function convertIndices(shape::Tuple, indices::Tuple{Vararg{CopyIndices}})
    println("enter enter: ", ntuple(i -> convertIndex(shape, i, indices[i]), length(indices)))
    return ntuple(i -> convertIndex(shape, i, indices[i]), length(indices))
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Bool})
    return findall(index)
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Int})
    println("Enter Top !! ", index)
    return index
end

function convertIndex(shape::Tuple, i::Int, index::Int)
    return [index]
end

function convertIndex(shape::Tuple, i::Int, index::Colon)
     println("Enter Bottom !! ", [x for x in 1:shape[i]])
    return [x for x in 1:shape[i]]
end

function convertIndex(shape::Tuple, i::Int, index::AbstractRange)
    println("Enter here !! ", collect(index))
    return collect(index)
end


function elementsAreValid(dims::Tuple, indices::Vararg{Int})
   return all((t) -> 1 <= t[1] <= t[2], zip(indices, dims))
end

function elementsAreValid(dims::Tuple, indices::Vararg{Union{Int,Colon}})
   return all((t) -> isa(t[1], Colon) || (1 <= t[1] <= t[2]), zip(indices, dims))
end
# Exceptions:
# ----------------

const DIMENSIONS_DO_NOT_MATCH = DomainError("The dimensions do not match ")
const INDICES_OUT_OF_BOUNDS = DomainError("The indices entered are outof bounds")
const SCALAR_ONLY_ACCESS_ERROR = DomainError("Cannot access non 0 dimensional array wiht array[()]")
const TYPE_NOT_ACCEPTED_AS_INDEX = DomainError("The type cannot be used to access elements in NDArray")
