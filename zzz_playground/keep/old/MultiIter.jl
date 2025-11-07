
"""
    MultiIter

An iterator that allows simultaneous iteration over multiple `Iter` objects (arrays), 
supporting broadcasting across different shapes.
"""
mutable struct MultiIter
    # Number of dimensions
    nd::Int

    # Number of iterators
    numiter::Int

    # Total number of elements
    size::Int

    # Actual index
    index::Int

    # dimensions also called shape
    dimensions::AbstractArray{Int}

    # array of iterators
    iters::AbstractArray{Iter}
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructor:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
    MultiIter(args...)

Construct a `MultiIter` from multiple arrays or existing `MultiIter` objects.  
Automatically handles broadcasting and prepares internal iterators for iteration.

# Complexity
- `O(n * nd)` where `n` is the total number of input arrays and `nd` is the maximum number of dimensions.
"""
function MultiIter(args...)

    # Initialize with default values
    multi =  MultiIter(0, 0, 0, 0, Int[], Iter[])

    # Fill the array of iterators
    foreach(arg -> _push_into!(arg, multi), args)

    # Verify that at least one iterator exist
    multi.numiter >= 1 || throw(DomainError("Wrong number of arguments"))

    _broadcast_multi(multi)

    # Reset the iterators to their start value
    _multi_iter_reset!(multi)

    # Return the ready to work multi iterator
    return multi
end

function _push_into!(mit::MultiIter, dest::MultiIter)
    for it in mit.iters
        push!(dest.iters, Iter( it.content ))
        dest.numiter += 1
    end
end

function _push_into!(obj, dest::MultiIter)
    push!(dest.iters, Iter(NDArray{eltype(obj)}(obj)))
    dest.numiter += 1
end

function _broadcast_multi(multi::MultiIter)
    # Fill the ndims, size and shape
    _fill_nd!(multi)
    _fill_size_and_dimensions!(multi)

    # Prepare the iterators to be iterated together 
    _prepare_iterators!(multi)
end

function _fill_nd!(mit::MultiIter)
    nd = 0
    for it in mit.iters
        nd = max(nd, Base.ndims(it.ao))
    end
    mit.nd = nd
end

function _fill_size_and_dimensions!(mit::MultiIter)
    nd = mit.nd
    
    # Initialize the dimensions
    mit.dimensions = Array{Int}(undef, nd)

    # Obtain the size of each dimension
    for i in 1:nd

        # By default the dimension is 1
        mit.dimensions[i] = 1

        # Obtain the broadcasted dimension
        for it in mit.iters

            # Compute which dimension of this array corresponds to the current broadcast dimension.
            # This handles arrays with fewer dimensions by "stretching" them on the left.
            src_dim = i + Base.ndims(it.ao) - nd

            # If the current array has a dimension corresponding to this broadcast axis:
            if (src_dim >= 0)

                # Obtain the size of that dimension in the array
                tmp = Base.size(it.ao, src_dim)

                # We have 3 scenarios:

                # 1. This array was "stretched" along this axis (size 1), so it can be broadcasted
                if (tmp == 1)    
                    continue
                # 2. The broadcast dimension is currently 1, so we can adopt the size of this array
                elseif (mit.dimensions[i] == 1)
                    mit.dimensions[i] = tmp

                 # 3. Sizes mismatch and neither is 1 → broadcasting is impossible
                elseif (mit.dimensions[i] != tmp)
                    throw(DomainError("shape mismatch exception"))
                end
            end
        end
    end

    # The total number of element in the array is the product of each dimensions
    mit.size = prod(mit.dimensions)
end

function _prepare_iterators!(mit::MultiIter)
    # Prepare each iterator in mit.iters so that they can be iterated together
    # according to the broadcast rules.
    for it in mit.iters
        
        # Set the total number of elements to match the broadcasted size
        it.size = mit.size

        # Number of dimensions in this array
        nd = Base.ndims(it.ao)

        # Initialize the last factor for linear index computation if array is not 0-d
        if (nd != 0)
            it.factors[mit.nd] = 1
        end

        # Loop over each broadcast dimension
        for j in eachindex(mit.nd)
            
            # dims_m1 stores the dimension size minus 1 (for coordinate calculations)
            it.dims_m1[j] = mit.dimensions[j] - 1

            # Map the current broadcast dimension to this array’s dimension
            k = j + nd - mit.nd

            # Determine strides: if array is smaller along this axis (or missing a dimension)
            if ((k < 1) || Base.size(it.ao)[k] != mit.dimensions[j])
                # Array is being broadcasted along this axis
                it.contiguous = false
                it.strides[j] = 0
            else
                # Otherwise the array has matching size → use its actual stride
                it.strides[j] = it.ao.strides[k]
            end
            
            # Compute backstrides used when resetting coordinates along this dimension
            it.backstrides[j] = it.strides[j] * it.dims_m1[j]

             # Precompute factors for linear index computation (multi-index → linear offset)
            if (j > 1)
                it.factors[mit.nd-j-1] = it.factors[mit.nd-j] * mit.dimensions[mit.nd-j]
            end
        end

        # Reset coordinates and pointer for iteration
        _iter_reset!(it)
    end
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Iterate:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
    Base.iterate(multi::MultiIter, state=nothing)

Iterates over the multi-iterator, returning the current values from all internal iterators.

# Returns
- `(vals::Vector, next_state::Int)` on each iteration.
- `nothing` when iteration is complete.

# Complexity
- `O(numiter)` per iteration.
"""
function Base.iterate(multi::MultiIter, state=nothing)
    # First iteration: reset the multi-iterator
    if state === nothing
        _multi_iter_reset!(multi)
        state = 1
    end

    # If we are past the total size, stop iteration
    if state > multi.size
        return nothing
    end

    # Advance the multi-iterator
    _multi_iter_next!(multi)

    # Get current values from all iterators
    vals = [it.data_ptr[] for it in multi.iters]

    # Return current values and next state
    return (vals, state + 1)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Next:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
    _multi_iter_next!(multi::MultiIter)

Advance all internal iterators of the `MultiIter` to the next element.  
Handles broadcasting automatically (stride 0 for repeated elements).

# Complexity
- `O(numiter * nd)` in the worst case (if all iterators are strided).
"""
function _multi_iter_next!(multi::MultiIter)

    # If the global iterator has already reached the end, stop iteration
    if multi.index >= multi.size
        return nothing
    end

    # Increment the global iteration index
    multi.index += 1

    # Advance each internal iterator to the next element
    # Handles broadcasting automatically (strides 0 for repeated elements)
    foreach(it -> _iter_next!(it), multi.iters)

    # Return the updated multi-iterator
    return multi
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Reset:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


"""
    _multi_iter_reset!(multi::MultiIter)

Reset the multi-iterator to the first element. Resets the global index and
all internal iterators.

# Complexity
- `O(numiter * nd)`
"""
function _multi_iter_reset!(multi::MultiIter)
    # Reset the global index to 0
    multi.index = 0
    # Reset each element of the arrays of iterator
    foreach(it -> _iter_reset!(it), multi.iters)
end






# function flattenIterator(mit::MultiIter, multi::MultiIter)
#     for j in eachindex(mit.numiter)
#         arr = mit.iters[j].content
#         push!(multi.iters, Iter( arr ))
#         multi.numiter += 1
#     end
# end