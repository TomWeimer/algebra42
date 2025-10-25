
"""
    Iter{DType}

An N-dimensional iterator over an `NDArray` (or compatible array).  
Supports linear and multi-dimensional indexing, broadcasting, and contiguous or strided access.

"""
mutable struct Iter{DType}
    # number of dimension - 1
    nd_m1::Int

    # number 
    index::Int

    # number of total element 
    size::Int

    # coordinates
    coordinates::AbstractArray{Int}

    # dimensions - 1
    dims_m1::AbstractArray{Int}

    # strides in bytes for each dimension
    strides::AbstractArray{Int}

    # used to reset the iterator at the end of a dimension
    backstrides::AbstractArray{Int}

    # shape factor
    factors::AbstractArray{Int}

    # actual array
    ao::AbstractArray{DType}

    # pointer on actual current item
    data_ptr::Ref{DType}

    contiguous::Bool

    bounds::NTuple{2,AbstractArray{Int}}

    limits::NTuple{2,AbstractArray{Int}}

    limits_sizes::AbstractArray{Int}
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructor:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #



"""
    Iter(ao::NDArray)
Constructs an `Iter` over the array `ao`. Initializes coordinates, strides, backstrides, bounds,
and other internal fields.

# Complexity
- `O(ndim)` where `ndim` is the number of dimensions of `ao`.
- Allocates `O(ndim)` arrays internally.
"""
function Iter(ao::NDArray)
    # Obtain diemension of the array
    nd = ndims(ao)

    # Intialize default values
    it = _init_iter(ao, nd)

    # Set factors if it is not a 0-dimensional array 
    if (nd != 0)
        it.factors[nd] = 1
    end

    # Fill the informations of the array
    for i in 1:nd
        dim = Base.size(ao, i)

        it.dims_m1[i] = dim - 1
        it.strides[i] = ao.strides[i]
        it.backstrides[i] = it.strides[i] * it.dims_m1[i]
        it.bounds[1][i] = 1
        it.bounds[2][i] = dim
        it.limits[1][i] = 1
        it.limits[2][i] = dim
        it.limits_sizes[i] = it.limits[2][i] - it.limits[1][i] + 1
    end
    return it
end

function _init_iter(ao::NDArray, nd::Integer)
    # Initialize with zero arrays
    zeros_nd() = zeros(Int, nd)

    return Iter(
        nd - 1,
        0,
        prod(size(ao)),
        zeros_nd(), zeros_nd(), zeros_nd(), zeros_nd(), zeros_nd(),
        ao,
        Ref(getItem(ao, 1)),
        true,
        (zeros_nd(), zeros_nd()),
        (zeros_nd(), zeros_nd()),
        zeros_nd()
    )
end

getItem(ao::NDArray{T, 0}) where T = ao[]

getItem(ao::NDArray{T, N}) where {T, N} = ao[1]

getItem(ao::NDArray{T, N}, i::Int) where {T, N} = ao[i]


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Next:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
    _iter_next!(it::Iter)

Advances the iterator `it` to the next element.

# Behavior
- Linear iteration for contiguous arrays.
- Multi-dimensional nested-loop style iteration for strided arrays.
- 0-dimensional arrays are handled as a special case.

# Returns
- Updated iterator `it` if not at the end.
- `nothing` if the iterator has reached the end.

# Complexity
- `O(1)` for contiguous arrays.
- `O(nd)` in the worst case for strided arrays (may need to carry over all dimensions).
"""
function _iter_next!(it::Iter)

    # If the iterator has already reached the end, stop iteration
    if it.index >= it.size
        return nothing
    end

    # Increment the linear (flattened) index
    it.index += 1

    # number of dimensions
    nd = length(it.coordinates)

    # If this is a 0-dimensional array, nothing else needs updating
    if nd == 0
        return it
    end

    # If the array is contiguous in memory, we can just move the pointer
    if it.contiguous
        it.data_ptr[] = getItem(it.ao, it.index)  # move pointer by 1 element
        return it
    end

    # Otherwise, handle multi-dimensional arrays
    # Update coordinates like nested loops (from last dimension to first)
    for d in nd:-1:1
        if it.coordinates[d] < it.dims_m1[d] + 1
            # Move forward in this dimension
            it.coordinates[d] += 1
             # Update the data pointer based on the multi-dimensional offset
            it.data_ptr[] = getItem(it.ao, _offset(it, nd))
            break # no need to update higher dimensions
        else
            # Reset this dimension and carry over to the next
            it.coordinates[d] = 1
            it.data_ptr[] = getItem(it.ao, _offset(it, nd))
        end
    end

    # Return the updated iterator
    return it
end

function _offset(it::Iter, nd::Integer)::Int
   offset = 0
   for i in 1:nd
      offset += (it.coordinates[i] - 1) * it.strides[i]
   end
   return offset + 1
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Reset:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
    _iter_reset!(it::Iter)

Resets the iterator `it` to the beginning.

- `index` is set to 0.
- All `coordinates` are reset to 0.

# Complexity
- `O(nd)` where `nd` is the number of dimensions.
"""
function _iter_reset!(it::Iter)
    it.index = 0
    it.coordinates = zeros(Int, Base.ndims(it.ao))
end
