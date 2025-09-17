
# Return a copy of the array with only the specified elements
const CopyIndices = Union{AbstractRange,Int,Colon,AbstractArray{Int},AbstractArray{Bool}}

# Type of the elements used in fancy indexing
const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# Exceptions:
# ----------------

const DIMENSIONS_DO_NOT_MATCH = DomainError("The dimensions do not match ")
const INDICES_OUT_OF_BOUNDS = DomainError("The indices entered are outof bounds")
const SCALAR_ONLY_ACCESS_ERROR = DomainError("Cannot access non 0 dimensional array wiht array[()]")
const TYPE_NOT_ACCEPTED_AS_INDEX = DomainError("The type cannot be used to access elements in NDArray")



# Padded Shape:
# ------------------

# Structure used when prepending shapes with ones
struct PaddedShape{OriginalDimension}
    originalShape::Ref{NTuple{OriginalDimension,Int}}
    expectedDimension::Ref{Int}
end

function PaddedShape(originalShape::NTuple{N,Int}, maxDim::Ref{Int}) where {N}
    # If the max dimension is smaller than the dimension of original shape then update it
    if (maxDim[] < N)
        maxDim[] = N
    end
    # Return a "view" on the original shape
    PaddedShape{N}(Ref(originalShape), maxDim)
end

function PaddedShape(originalArray::AbstractArray{T,N}, maxDim::Ref{Int}) where {T,N}
    # If the max dimension is smaller than the dimension of the original array then update it
    if (maxDim[] < N)
        maxDim[] = N
    end
    # Return a "view" on the shape of the array
    PaddedShape{N}(Ref(size(originalArray)), maxDim)
end

# Using this function and an offset we mimic a shape prepended with 1 without allocating it
function Base.getindex(prep_shape::PaddedShape{OriginalDimension}, i::Int) where {OriginalDimension}

    if (prep_shape.expectedDimension[] == OriginalDimension)
        return prep_shape.originalShape[][1]
    else
        offset = prep_shape.expectedDimension[] - OriginalDimension
        #println("ExpectedDimension: ", prep_shape.expectedDimension[], " OriginalDimension: ", OriginalDimension, "  offset: ", offset, "  i: ", i)
        return (i <= offset) ? 1 : prep_shape.originalShape[][i-offset]
    end
end

# Return the original dimension of the shape
original_dim(padded_shape::PaddedShape{OriginalDimension}) where {OriginalDimension} = OriginalDimension

Base.length(padded_shape::PaddedShape{OriginalDimension}) where {OriginalDimension} = padded_shape.expectedDimension[]

# Acess:
#-------

# Access to fancy indexing using directly only one index
function Base.getindex(array::NDArray{DType,N}, indices::CopyIndices) where {DType,N}
    shape = size(array)
    convertedIndices = convertIndices(shape, indices)
    return fancyIndexing(array, convertedIndices)
end

# Access to fancy indexing using directly multiple index
function Base.getindex(array::NDArray{DType,N}, indices::Vararg{CopyIndices}) where {DType,N}
    shape = size(array)
    convertedIndices = convertIndices(shape, indices)
    return fancyIndexing(array, convertedIndices)
end

# Access to fancy indexing using directly only one index
function Base.getindex(array::NDArray{DType,N}, indices::AbstractArray{Bool,1}) where {DType,N}
    return fancyIndexing(array, indices)
end

# Process:
#---------

# A single bool mask
function fancyIndexing(array::NDArray{DType,N}, mask::AbstractArray{Bool}) where {DType,N}
    # Verify that they have the same number of elements
    length(mask) == array.shape.length || throw(DIMENSIONS_DO_NOT_MATCH)

    # Create the array if the boolean was true
    filteredArray = [val for (i, val) in enumerate(array) if mask[i] == true]

    # Return a new NDArray
    return NDArray{DType}(filteredArray)
end

# Multiple index, elements are all continuous
function fancyIndexing(original_array::NDArray{DType,N}, indices::Tuple{Vararg{AbstractArray{Int}}}) where {DType,N}
  #  println("indices: ", indices)

    output_shape, broadcast_function = advanced_indexing(indices) do output_idx::CartesianIndex
        #println("indices: ", indices, " output_idx: ", output_idx, " length: ", length(indices))
        src_idx = ntuple(j -> length(indices[j]) == 1 ? 1 : indices[j][output_idx[j] ], length(indices))
      #  println("indices: ", indices, " sourceIdx: ", src_idx)
        return src_idx
    end

    # println("output_shape: ", output_shape)

    output_array = NDArray{DType}(output_shape)

    #println("output_shape: ", output_shape)

    for i in CartesianIndices(output_shape)
        found_index = broadcast_function(i)
    #    println("i: ", found_index)
        output_array[i] = original_array[found_index...]
    end

    # Case 1: Scalar
    if all(x -> x == 1, output_shape)
        return output_array[1]   # return the scalar directly
    end

   # println("arrived here", "output_shape: ", output_shape)
    # Case 2: Drop singleton dims
    new_shape = filter(!=(1), output_shape)
    #  println("enter here")
    if length(new_shape) < length(output_shape)
        reshape(output_array, new_shape...)
    end

    # Case 3: Normal case
    return output_array
end




# Broadcasting advanced indexing:
# -------------------------------

function advanced_indexing(f::Function, index::AbstractArray{Int})
    return advanced_indexing((index,), f)
end

function advanced_indexing(f::Function, indices::Tuple{Vararg{AbstractArray{Int}}})
    # Broadcasting can be understood by four rules
    #   1. All input arrays with ndim smaller than the input array of largest ndim have
    #      1’s pre-pended to their shapes.
    #
    #   2. The size in each dimension of the output shape is the maximum of all the
    #      input shapes in that dimension.
    #
    #   3. An input can be used in the calculation if it’s shape in a particular dimension
    #      either matches the output shape or has value exactly 1.
    #
    #   4. If an input has a dimension size of 1 in its shape, the first data entry in that
    #      dimension will be used for all calculations along that dimension. In other
    #      words, the stepping machinery of the ufunc will simply not step along that
    #      dimension when otherwise needed (the stride will be 0 for that dimension).

    # Step 1: Prepend input arrays with ones

    padded_shapes = prependInputArraysShape(indices)

    # Step 2: Obtain output shape

    N = length(padded_shapes[1])

    output_shape = obtain_output_shape(padded_shapes, N)

    # Step 3: Either the input can be used as a "broadcast constant" or we use it as a "variable index"
    is_broadcast_compatible(output_shape, padded_shapes, N) || throw(DomainError("The indices entered are not compatible for broadcasting"))

    # Step 4: Use broadcast constant constant when shape is 1 in a dimension
    broadcast_constant_index, broadcast_constant_values = obtain_broadcast_constant(padded_shapes::AbstractArray{<:PaddedShape}, indices, N::Int)

    # Create and return a function to be broadcast on the original_array 
    return output_shape, function projected_f(input_index::CartesianIndex)
        # Create a full CartesianIndex for the original function
        output_index = zeros(Int, N == 1 ? length(indices) : N)


        # println("broadcast constant: ", broadcast_constant_values)
        # Fill the output index with the broadcast constant
        for i in eachindex(broadcast_constant_index)
            output_index[broadcast_constant_index[i]...] = broadcast_constant_values[i]
        end

        #   println("N: ", N, "input_index: ", input_index, "output_idx: ", output_index)

        # Fill the output index with the non constant values
        for (i, val) in enumerate(output_index)
            if (val == 0)
                output_index[i] = input_index[i]
            end
        end

        #  println("input_index: ", input_index, "output_idx: ", output_index)

        # Call the original function with the full CartesianIndex
        return f(CartesianIndex(Tuple(output_index)))
    end
end

# Step 1
function prependInputArraysShape(indices::Tuple{Vararg{AbstractArray{Int}}})
    maxDim = 0
    ref_maxDim = Ref(maxDim)
    return [PaddedShape(index, ref_maxDim) for index in indices]
end

# Step 2
function obtain_output_shape(padded_shapes::AbstractArray{<:PaddedShape}, N::Int)
    return (N == 1) ? ntuple(i -> padded_shapes[i][0], length(padded_shapes)) : ntuple(i -> maximum(ps[i] for ps in padded_shapes), N)
end

# Step 3
function is_broadcast_compatible(output_shape::Tuple, padded_shapes::AbstractArray{<:PaddedShape}, N::Int)
    return (N == 1) ? all(i -> length(padded_shapes[i]) == 1, length(padded_shapes)) : all(i -> all(ps[i] == 1 || ps[1] == output_shape[i] for ps in padded_shapes), 1:N)
end

# Step 4
function obtain_broadcast_constant(padded_shapes::AbstractArray{<:PaddedShape}, original_indices::Tuple{Vararg{AbstractArray{Int}}}, N::Int)
    broadcast_constant_index = Vector{NTuple{N,Int}}()
    broadcast_constant_values = Int[]

    for dim in 1:N
        idx_array = original_indices[dim]
        shape_dim = padded_shapes[dim]

        for i in eachindex(idx_array)
            # Check if this dimension was broadcasted
            if shape_dim[i] == 1
                # Create a full N-tuple with dim index replaced by i, others can be left as 1 or 0
                index_tuple = ntuple(d -> d == dim ? i : 1, N)
                push!(broadcast_constant_index, index_tuple)
                push!(broadcast_constant_values, idx_array[i])
            end
        end
    end
    return broadcast_constant_index, broadcast_constant_values
end

function convertIndices(shape::Tuple, index::CopyIndices)
    return (convertIndex(shape, 1, index),)
end

function convertIndices(shape::Tuple, indices::Tuple{Vararg{CopyIndices}})
    return ntuple(i -> convertIndex(shape, i, indices[i]), length(indices))
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Bool})
    return findall(index)
end

function convertIndex(shape::Tuple, i::Int, index::AbstractArray{Int})
    return index
end

function convertIndex(shape::Tuple, i::Int, index::Int)
    return [index]
end

function convertIndex(shape::Tuple, i::Int, index::Colon)
    return [x for x in 1:shape[i]]
end

function convertIndex(shape::Shape, i::Int, index::AbstractRange)
    return collect(index)
end


