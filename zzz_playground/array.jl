# Test for 3D array  

struct Shape{N}
    length::Int              # size
    dims::Tuple{Vararg{Int}} # shape
    
    # Inner constructor for scalars
    Shape{0}() = new{0}(1, (0,))

    # Inner constructor for a general Shape from a tuple of dimensions
    function Shape{N}(tuple::NTuple{N, Int}) where {N}
        if any(d -> d < 0, tuple)
            throw(DomainError("Dimensions cannot be negative"))
        end
        new{N}(prod(tuple), tuple)
    end
end


function _compute_strides(shape::Shape)
    dim = length(shape.dims)
    # Pre-allocate a tuple of zeros
    strds = zeros(Int, dim)
    # The stride for the first dimension is always 1
    strds[1] = 1
    # For subsequent dimensions, stride[i] = stride[i-1] * size[i-1]
    for i in 2:dim
        strds[i] = strds[i-1] * shape.dims[i-1]
    end
    return Tuple(strds)
end


# Obtain the index ( flat index ) from cartesian index
function _offset(strides::Tuple{Vararg{Int}}, shape::Shape{N}, I::CartesianIndex)::Int where {N}
   idx_tuple = Tuple(I)
   offset = 0;
   for i in 1:N
       offset += (idx_tuple[i] - 1) * strides[i]  # subtract 1 because Julia indices are 1-based
   end
   return offset + 1;
end


shape_3d = Shape{3}((2, 2, 2))
strides_3d = _compute_strides(shape_3d)
println("3D Strides: ", strides_3d) # Should be (1, 2, 4)

# Test indices for 3D array
test_indices_3d = [
    CartesianIndex(1, 1, 1), # Should be 1 (value 1)
    CartesianIndex(2, 1, 1), # Should be 2 (value 3)
    CartesianIndex(1, 2, 1), # Should be 3 (value 2)
    CartesianIndex(2, 2, 1), # Should be 4 (value 4)
    CartesianIndex(1, 1, 2), # Should be 5 (value 5)
    CartesianIndex(2, 1, 2), # Should be 6 (value 7)
    CartesianIndex(1, 2, 2), # Should be 7 (value 6)
    CartesianIndex(2, 2, 2)  # Should be 8 (value 8)
]

for idx in test_indices_3d
    flat_idx = _offset(strides_3d, shape_3d, idx)
    println("Index $idx -> Flat index: $flat_idx")
end


# Get the actual memory layout by converting to a vector
A_3d = [1 2; 3 4 ;;; 5 6; 7 8]
actual_memory = vec(A_3d)
println("Actual Julia memory layout: ", actual_memory)
# Should be: [1, 3, 2, 4, 5, 7, 6, 8]

# Compare with our calculated indices
for idx in test_indices_3d
    flat_idx = _offset(strides_3d, shape_3d, idx)
    calculated_value = actual_memory[flat_idx]
    actual_value = A_3d[idx]
    println("Index $idx -> Flat: $flat_idx, Calculated: $calculated_value, Actual: $actual_value, Match: $(calculated_value == actual_value)")
end

