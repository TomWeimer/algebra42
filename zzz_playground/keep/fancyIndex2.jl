

# # Index types:
# # -----------

# # Return a single elements of the array ( if enough are provided )
# const ElementIndex = Int                          

# # Return a view if:
# # - a colon is present
# # - the number of indices given is less than the number of array's dimensions
# const ViewIndices = Union{AbstractRange, Int, Colon} 

# # Fancy indices are arrays of integer or boolean
# const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# # Return a copy of the array, if at least one fancy index is present
# const CopyIndices = Union{AbstractRange, Int, Colon, AbstractArray{Int}, AbstractArray{Bool}}



# # Accessing 0-dimensional array:
# # -----------------------------

# """
#     Base.getindex(array::NDArray{DType,0}, ::Tuple{}) where {DType}

# Returns the single element from a 0-dimensional `NDArray`.

# A 0-dimensional ndarray represents a single, scalar value. According to Julia's
# indexing rules, the correct way to retrieve this element is by using `array[]`.
# This method enable the numpy's syntax `array[()]` if the array is 0-dimensional.
# """
# function Base.getindex(array::NDArray{DType,0}, ::Tuple{}) where {DType}
#     return item(array)
# end

# """
#     Base.getindex(::NDArray{DType,0}, indices::Int) where {DType}

# Throw an error if we try to access a 0-dimensional ndarrray with the syntax `array[1]`

# """
# function Base.getindex(::NDArray{DType,0}, indices::Int) where {DType}  # return an error
#     throw(DomainError("accessing by index with a scalar is only possible as ndarray[()] or ndarray.item()"))
# end


# """
#     Base.getindex(::NDArray{DType, N}, args::Tuple{}) where {DType, N}

# Prevents the use of a 0-dimensional index `array[()]` on a multi-dimensional `NDArray`.

# This method serves as a safety check, explicitly throwing a `DomainError` when a user
# attempts to access a non-scalar array with an empty tuple. This syntax is
# reserved for 0-dimensional arrays to retrieve their single element.

# The function correctly identifies that `N > 0` and provides a clear error message
# explaining why this operation is invalid for multi-dimensional arrays.
# """
# function Base.getindex(::NDArray{DType,N}, args::Tuple{}) where {DType,N} # return an error
#     # Check that the tuple is not empty
#     isempty(args) && throw(DomainError("The dimension is not 0 ndarray[()] works only on scalar"))
#     # Throw an error the element can not be accessed with a tuple
#     throw(DomainError("The element cannnot be accessed with tuple"))
# end


# """
#     Base.getindex(::NDArray{DType,0}, ::Vararg{ViewIndices}) where {DType, N}

# Prevents the use of slices in 0-dimensional `NDArray` such as `array[:]`.

# This method serves as a safety check, explicitly throwing a `DomainError` when a user
# attempts to slice the 0-dimensional array.
# """
# Base.getindex(::NDArray{DType,0}, ::Vararg{ViewIndices}) where {DType} = throw(DomainError("Cannot access to scalar with ndarray[:]"))



# # Accessing N-dimensional array:
# # -----------------------------

# """
#     Base.getindex(array::NDArray{DType, N}, indices::Vararg{ElementIndex, N}) where {DType, N}

# Returns a single element from an N-dimensional `NDArray` using a fixed number of integer indices.

# This method overloads the `getindex` function to enable direct element access using a
# tuple of indices, such as `my_array[1, 2, 3]`. It is a fundamental part of the
# NDArray's API, providing the standard Julia-like indexing behavior.

# # Arguments
# - `array::NDArray{DType, N}`: The N-dimensional array from which to retrieve an element.
# - `indices::Vararg{ElementIndex, N}`: A variadic tuple of integer indices. The
#   number of indices provided must match the dimension `N` of the array.

# # Returns
# - A single element of type `DType` at the specified index.

# # Throws
# - `DomainError`: Thrown if the provided indices are out of bounds or if the
#   number of indices does not match the array's dimension `N`.
# """
# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element

#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

#     # Check if the index are valid
#     (elementsAreValid(size(array), indices...)) || throw(DomainError("Indexes out of bounds"))

#     # If the index is valid return the associated elements
#     return array.content[_offset(array, indices...)]
# end


# """
#     Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}

# Returns a view or a slice of an `NDArray` using a partial set of integer indices.

# This method overloads `getindex` to provide flexible array slicing, similar to Python's NumPy.
# If the number of indices provided is less than the dimension `N` of the array,
# the remaining dimensions are implicitly treated as full slices (`:`). This allows for
# simple operations like `my_array[1, 2]` to return a sub-array (a view) rather than throwing an error.

# The function works by "stretching" the provided indices into a full `N`-element tuple
# by appending a `Colon()` object for each missing index. This new tuple is then
# used to perform the actual slicing operation on the underlying data. 

# # Arguments
# - `array::NDArray{DType,N}`: The multi-dimensional array to be sliced.
# - `indices::Vararg{ElementIndex}`: A variadic tuple of integer indices. The number
#   of indices can be from 1 up to the dimension `N`.

# # Returns
# - A view on the orignal `NDArray`.

# # Throws
# - `DomainError`: Thrown if the number of indices provided is greater than the
#   array's dimension `N`, or if the indices are empty.
# """
# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))

#     # Check if the number of index is valid
#     dimIndices, dimArray = length(indices), ndims(array)

#     (dimIndices <= dimArray) || throw(DomainError("The number of index provided is too big"))

#     indicesStretched = (dimIndices == dimArray) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimArray)

#     # If the index is valid return the associated elements
#     return array[indicesStretched...]
# end


# """
#     Base.getindex(array::NDArray{DType, N}, indices::Vararg{ViewIndices}) where {DType, N}

# Performs array slicing and returns a memory-efficient `view` of the `NDArray`.

# This method enables advanced indexing using `AbstractRange` (e.g., `1:3`) and `Colon` (`:`).
# Instead of creating a full copy of the data, it returns a lazy `view` into the original array. 
# This makes slicing operations very fast and memory-efficient, especially for large arrays.

# # Arguments
# - `array::NDArray{DType, N}`: The N-dimensional array to be sliced.
# - `indices::Vararg{ViewIndices}`: A variadic tuple of indices. `ViewIndices`
#   can be a mix of `Int`, `Colon`, and `AbstractRange`.

# # Returns
# - A new `NDArray` object representing a `view` of the original array.

# # Throws
# - `DomainError`: Thrown if the provided indices are out of bounds.
# """
# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{ViewIndices}) where {DType,N}
#     # Check that the indices are not empty
#     isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
#     # Check if the index are valid
#     #@infiltrate
#     _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
#     # If the index is valid return the associated elements
#     return @view array[indices...]
# end


# # special indices:
# # ----------------

# """
#     Base.getindex(A::NDArray, I::CartesianIndex)

# Returns a single element from an `NDArray` using a `CartesianIndex` object.

# A `CartesianIndex` is a type that holds a multi-dimensional index. It is commonly
# used when iterating over the elements of a multi-dimensional array, as it provides
# a convenient way to represent each element's position. This method overloads `getindex`
# to allow direct element access using this type.

# # Arguments
# - `A::NDArray`: The N-dimensional array from which to retrieve the element.
# - `I::CartesianIndex`: The `CartesianIndex` object specifying the position of the
#   desired element.

# # Returns
# - A single element from the array at the position specified by `I`.
# """
# function Base.getindex(A::NDArray{T, N}, I::CartesianIndex{N}) where {T, N}
#     A.content[_offset(A, I)]
# end

# # fancy indices:
# # ----------------

# """
#     Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}

# Performs "fancy indexing" on an `NDArray` using a single index, such as a boolean array or a vector of integers.

# This method overloads `getindex` to provide a powerful and flexible way to select
# multiple non-contiguous elements from an array. It delegates the complex logic of
# fancy indexing to a separate `fancyIndexing` function, which handles the
# conversion and broadcasting of indices.

# # Arguments
# - `array::NDArray{DType,N}`: The N-dimensional array from which to select elements.
# - `indices::CopyIndices`: A single fancy index, which can be an array of integers,
#   a boolean array, or any type that represents a collection of indices.

# # Returns
# - A new `NDArray` containing the selected elements. This is a copy of the data,
#   not a view.

# # Implementation Details
# - The function first converts the input `indices` into a standardized format
#   using the `convertIndices` helper function.
# - It then calls `fancyIndexing` to perform the actual element selection and
#   return the new `NDArray`. 

# # Example
# ```julia
# nd_array = NDArray([10, 20, 30, 40, 50])

# # Select elements at index 1 and 3
# subset_vector = nd_array[[1, 3]] 
# # subset_vector will be a new NDArray with content [10, 30]

# # Select elements based on a boolean mask
# bool_mask = [true, false, true, false, false]
# subset_bool = nd_array[bool_mask]
# # subset_bool will be a new NDArray with content [10, 30]
# """
# function Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}
#     shape = size(array)
#     convertedIndices = convertIndices(shape, indices)
#     return fancyIndexing(array, convertedIndices)
# end


# """
#     Base.getindex(array::NDArray{DType, N}, indices::Vararg{CopyIndices}) where {DType, N}

# Performs advanced indexing or fancy indexing on an `NDArray` using a combination of multiple indices.

# This method handles complex indexing scenarios where a user provides multiple
# indices, such as a mix of vectors and boolean masks. The function is designed to
# be the entry point for all advanced indexing operations.

# It works by delegating the heavy lifting to two helper functions: `convertIndices`,
# which standardizes the input indices, and `fancyIndexing`, which performs the
# actual selection and broadcasting logic to create the final result. 

# # Arguments
# - `array::NDArray{DType, N}`: The N-dimensional array from which to select elements.
# - `indices::Vararg{CopyIndices}`: A variadic tuple of fancy indices. This can be
#   a mix of arrays of integers, boolean arrays, or other indexable types.

# # Returns
# - A new `NDArray` containing the selected elements. This is a copy of the data,
#   not a view.
# """
# function Base.getindex(array::NDArray{DType,N}, indices::Vararg{CopyIndices}) where {DType,N}
#    # println("enter motherfucker2")
#     shape = size(array)
#     convertedIndices = convertIndices(shape, indices)
#    # println("indices: ", indices)
#     return fancyIndexing(array, convertedIndices)
# end



# # Process:
# #---------

# """
#     fancyIndexing(array::NDArray{DType,1}, mask::AbstractArray{Bool}) where {DType,N}

# Performs boolean mask indexing on an `NDArray`.

# This method allows a user to select elements from an array based on a boolean mask.
# The mask must be the same size as the array, with `true` values indicating which
# elements to include in the result and `false` values indicating elements to exclude.

# The function iterates through the elements of the original array and the boolean
# mask simultaneously, building a new, 1-dimensional array that contains only the
# elements corresponding to `true` values in the mask. This is a common and powerful
# technique for data filtering and analysis. 

# # Arguments
# - `array::NDArray{DType, 1}`: The source array to be indexed.
# - `mask::AbstractArray{Bool}`: A boolean array of the same size as `array`.

# # Returns
# - A new 1-dimensional `NDArray` containing the elements that correspond to `true`
#   in the mask.

# # Throws
# - `DimensionError`: Thrown if the `mask` and the `array` do not have the same number
#   of elements.

# # Example
# ```jldoctest
# julia> nd_array = NDArray([10 20 30 40 50 60])

# # A boolean mask for a 1D array
# julia> mask = [true false true false true false]

# # Select elements based on the mask
# julia> filtered_array = fancyIndexing(nd_array, mask)
# # filtered_array will be a new NDArray with content [10, 30, 50]
# ```
# """
# function fancyIndexing(array::NDArray{DType, 1}, mask::AbstractArray{Bool}) where {DType}
#     # Verify that they have the same number of elements
#     length(mask) == array.shape.length || throw(DIMENSIONS_DO_NOT_MATCH)

#     # Create the array if the boolean was true
#     filteredArray = [val for (i, val) in enumerate(array) if mask[i] == true]

#     # Return a new NDArray
#     return NDArray{DType}(filteredArray)
# end

# """
#     fancyIndexing(original_array::NDArray{DType,N}, indices::Tuple{Vararg{AbstractArray{Int}}}) where {DType,N}

# Performs advanced indexing on an `NDArray` using a tuple of integer arrays, returning a new, contiguous array.

# This method handles "fancy indexing" where multiple dimensions are indexed by arrays
# of integers. It is designed to emulate the behavior of libraries like NumPy, where
# the resulting array's shape is determined by the broadcasting of the index arrays.

# The function first calculates the `output_shape` and a `broadcast_function` that maps
# indices from the output array back to the original array. This allows it to efficiently
# build the new array by iterating over its indices and retrieving the correct elements
# from the source array.

# # Arguments
# - `original_array::NDArray{DType,N}`: The source N-dimensional array.
# - `indices::Tuple{Vararg{AbstractArray{Int}}}`: A tuple where each element is an array
#   of integers.

# # Returns
# - A new `NDArray` containing the selected elements. This is a copy of the data, not a view.
#   - If the result is a scalar, it returns the scalar value directly.
#   - If the result has singleton dimensions (e.g., `(1, 4, 1)`), it returns a reshaped array
#     with those dimensions dropped (e.g., `(4,)`).
#   - Otherwise, it returns the new array as-is.

# # Example
# ```jldoctest
# nd_array = NDArray(reshape(collect(1:12), 3, 4))

# # Select elements at row 1, column 2 and row 3, column 4
# subset = fancyIndexing(nd_array, ([1, 3], [2, 4]))
# # The result will be a 2-element 1D NDArray with content [2, 12]
# ````
# """
# function fancyIndexing(original_array::NDArray{DType,N}, indices::Tuple{Vararg{AbstractVector{Int}}}) where {DType,N}
#     output_shape, broadcast_function = advanced_indexing(indices) do output_idx::CartesianIndex
#         src_idx = ntuple(length(indices)) do j
#              return length(indices[j]) == 1 ? 1 : indices[j][output_idx[j]]
#         end
#         return src_idx
#     end

#     output_array = NDArray{DType}(output_shape)

#     for i in CartesianIndices(output_shape)
#         found_index = broadcast_function(i)
#         output_array[i] = original_array[found_index...]
#     end

#     # Case 1: Scalar
#     if all(x -> x == 1, output_shape)
#         return output_array[1]   # return the scalar directly
#     end

#     # Case 2: Drop singleton dims
#     new_shape = filter(!=(1), output_shape)
#     if length(new_shape) < length(output_shape)
#         reshape(output_array, new_shape...)
#     end

    

#     # Case 3: Normal case
#     return output_array
# end

# # TODO:
# # function fancyIndexing(original_array::NDArray{DType,N}, indices::Tuple{Vararg{AbstractArray{Int}}}) where {DType,N}

# #         println("enter in fatherfucker")


# #     output_shape, broadcast_function = advanced_indexing(indices) do output_idx::CartesianIndex
# #         src_idx = ntuple(length(indices)) do j
# #             inds = indices[j]
# #             if MyMath.prod(size(inds)) == 1
# #                 inds[]
# #             else
# #                 inds[output_idx]
# #             end
# #         end
# #         return src_idx
# #     end

# #     output_array = NDArray{DType}(output_shape)

# #     for i in CartesianIndices(output_shape)
# #         found_index = broadcast_function(i)
# #         output_array[i] = original_array[found_index...]
# #     end

# #     # Case 1: Scalar
# #     if all(x -> x == 1, output_shape)
# #         return output_array[1]   # return the scalar directly
# #     end

# #     # Case 2: Drop singleton dims
# #     new_shape = filter(!=(1), output_shape)
# #     if length(new_shape) < length(output_shape)
# #         reshape(output_array, new_shape...)
# #     end

    

# #     # Case 3: Normal case
# #     return output_array
# # end


# # Broadcasting advanced indexing:
# # -------------------------------

# function advanced_indexing(f::Function, index::AbstractArray{Int})
#     return advanced_indexing((index,), f)
# end

# function advanced_indexing(f::Function, indices::Tuple{Vararg{AbstractArray{Int}}})
#     # Broadcasting can be understood by four rules
#     #   1. All input arrays with ndim smaller than the input array of largest ndim have
#     #      1’s pre-pended to their shapes.
#     #
#     #   2. The size in each dimension of the output shape is the maximum of all the
#     #      input shapes in that dimension.
#     #
#     #   3. An input can be used in the calculation if it’s shape in a particular dimension
#     #      either matches the output shape or has value exactly 1.
#     #
#     #   4. If an input has a dimension size of 1 in its shape, the first data entry in that
#     #      dimension will be used for all calculations along that dimension. In other
#     #      words, the stepping machinery of the ufunc will simply not step along that
#     #      dimension when otherwise needed (the stride will be 0 for that dimension).

#     # Step 1: Prepend input arrays with ones

#     padded_shapes = prependInputArraysShape(indices)

#     # Step 2: Obtain output shape

#     N = length(padded_shapes[1])

#     output_shape = obtain_output_shape(padded_shapes, N)




#     # Step 3: Either the input can be used as a "broadcast constant" or we use it as a "variable index"
#     is_broadcast_compatible(output_shape, padded_shapes, N) || throw(DomainError("The indices entered are not compatible for broadcasting"))

#     #println("N: ", N, " padded shapes: ", size.(padded_shapes), " output_shape: ", output_shape, "is broadcast compatible: ", is_broadcast_compatible(output_shape, padded_shapes, N))

#     # Step 4: Use broadcast constant constant when shape is 1 in a dimension
#     broadcast_constant_index, broadcast_constant_values = obtain_broadcast_constant(padded_shapes::AbstractArray{<:PaddedShape}, indices, N::Int)

#     # Create and return a function to be broadcast on the original_array 
#     return output_shape, function projected_f(input_index::CartesianIndex)
#         # Create a full CartesianIndex for the original function
#         output_index = zeros(Int, N == 1 ? length(indices) : N)

#         # Fill the output index with the broadcast constant
#         for i in eachindex(broadcast_constant_index)
#             output_index[broadcast_constant_index[i]...] = broadcast_constant_values[i]
#         end

#         # Fill the output index with the non constant values
#         for (i, val) in enumerate(output_index)
#             if (val == 0)
#                 output_index[i] = input_index[i]
#             end
#         end

#         # Call the original function with the full CartesianIndex
#         return f(CartesianIndex(Tuple(output_index)))
#     end
# end

# # Step 1
# function prependInputArraysShape(indices::Tuple{Vararg{AbstractArray{Int}}})
#     maxDim = 0
#     ref_maxDim = Ref(maxDim)
#     return [PaddedShape(index, ref_maxDim) for index in indices]
# end

# # Step 2
# function obtain_output_shape(padded_shapes::AbstractArray{<:PaddedShape}, N::Int)
#     return (N == 1) ? ntuple(i -> padded_shapes[i][0], length(padded_shapes)) : ntuple(i -> maximum(ps[i] for ps in padded_shapes), N)
# end

# # Step 3
# function is_broadcast_compatible(output_shape::Tuple, padded_shapes::AbstractArray{<:PaddedShape}, N::Int)
#     return (N == 1) ? all(i -> length(padded_shapes[i]) == 1, length(padded_shapes)) : all(i -> all(ps[i] == 1 || ps[i] == output_shape[i] for ps in padded_shapes), 1:N)
# end

# # Step 4
# function obtain_broadcast_constant(padded_shapes::AbstractArray{<:PaddedShape}, original_indices::Tuple{Vararg{AbstractArray{Int}}}, N::Int)
#     broadcast_constant_index = Base.Vector{NTuple{N,Int}}()
#     broadcast_constant_values = Int[]

#     for dim in 1:N
#         idx_array = original_indices[dim]
#         shape_dim = padded_shapes[dim]

#         for i in eachindex(idx_array)
#             # Check if this dimension was broadcasted
#             if shape_dim[i] == 1
#                 # Create a full N-tuple with dim index replaced by i, others can be left as 1 or 0
#                 index_tuple = ntuple(d -> d == dim ? i : 1, N)
#                 push!(broadcast_constant_index, index_tuple)
#                 push!(broadcast_constant_values, idx_array[i])
#             end
#         end
#     end
#     return broadcast_constant_index, broadcast_constant_values
# end

# # Internal methods:
# # -----------------

# function convertIndices(shape::Tuple, index::CopyIndices)
#     return (convertIndex(shape, 1, index),)
# end

# function convertIndices(shape::Tuple, indices::Tuple{Vararg{CopyIndices}})
#     return ntuple(i -> convertIndex(shape, i, indices[i]), length(indices))
# end

# function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Bool})
#     return findall(index)
# end

# function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Int})
#     return index
# end

# function convertIndex(shape::Tuple, i::Int, index::Int)
#     return [index]
# end

# function convertIndex(shape::Tuple, i::Int, index::Colon)
#     return [x for x in 1:shape[i]]
# end

# function convertIndex(shape::Shape, i::Int, index::AbstractRange)
#     return collect(index)
# end


# function elementsAreValid(dims::Tuple, indices::Vararg{Int})
#    return all((t) -> 1 <= t[1] <= t[2], zip(indices, dims))
# end

# function elementsAreValid(dims::Tuple, indices::Vararg{Union{Int,Colon}})
#    return all((t) -> isa(t[1], Colon) || (1 <= t[1] <= t[2]), zip(indices, dims))
# end

# # Exceptions:
# # ----------------

# const DIMENSIONS_DO_NOT_MATCH = DomainError("The dimensions do not match ")
# const INDICES_OUT_OF_BOUNDS = DomainError("The indices entered are outof bounds")
# const SCALAR_ONLY_ACCESS_ERROR = DomainError("Cannot access non 0 dimensional array wiht array[()]")
# const TYPE_NOT_ACCEPTED_AS_INDEX = DomainError("The type cannot be used to access elements in NDArray")

# # # Access to fancy indexing using directly only one index
# # function Base.getindex(array::NDArray{DType,N}, indices::AbstractArray{Bool,1}) where {DType,N}
# #     return fancyIndexing(array, indices)
# # end


# # abstract type AbstractNDArray end



# struct ColVector{T} <: AbstractMatrix{T}
#     parent::NDArray{T, 1}
# end


# struct RowVector{T} <: AbstractMatrix{T}
#     parent::NDArray{T, 1}
# end

# ColVector(v::Vector{T}) where {T} = ColVector(v)
# RowVector(v::Vector{T}) where {T} = ColVector(v)


