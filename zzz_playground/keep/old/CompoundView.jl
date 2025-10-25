include("NDArrayView.jl")

struct CompoundView{T, N, SegmentNB} <: AbstractNDArray{T, N}
    axis::Int
    segments::NTuple{SegmentNB, AbstractArray{T, N}}
    segment_ranges::NTuple{SegmentNB, OrdinalRange{Int, Int}}
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructor                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Concatenate multiple array together:
# ---------------------------------------------

function concatenate(axis::Int, segments::Vararg{AbstractArray{DType, NDim}, SegmentNB}) where {DType, NDim, SegmentNB}

    shapes = size.(segments)

    #println("segments: $segments, axis: $axis, shapes: $shapes")
    
    areConcatenable(shapes, axis) || throw( DimensionMismatch("Those arrays have not the same size along $axis") )

   
    shapes_along_axis = getindex.(shapes, axis)

    shape = obtainConcatShape(shapes, shapes_along_axis, axis)
    
    segment_ranges = obtainSegmentRanges(shapes_along_axis)
    return dropdims(CompoundView{DType, length(shape), length(segments)}(shape, segments, segment_ranges, axis); dims=nothing)
end

# The concatenation affects only one dimension, whose length equals the sum of the input arrays along that axis.
function obtainConcatShape(shapes::NTuple{SegmentNB, NTuple{NDim, Int}}, shapes_along_axis::Tuple, axis::Int) where {SegmentNB, NDim}
    ref_shape = first(shapes)
    new_dim =  sum(shapes_along_axis)
    return Tuple((i == axis) ? new_dim : ref_shape[i]  for i in 1:NDim )
end

# We define index ranges to map each input index to its corresponding segment.
function obtainSegmentRanges(shapes_along_axis::Tuple)
    
    N = length(shapes_along_axis)

    segment_ranges = OrdinalRange{Int, Int}[]

    push!(segment_ranges, 1:first(shapes_along_axis))

    for i in 2:N
        start = last(segment_ranges[i - 1]) + 1
        stop = start + shapes_along_axis[i] - 1
        push!(segment_ranges, start:stop)
    end

    return tuple(segment_ranges...)
end

# Create a trivial view of a compound view:
# -----------------------------------------

create_view_from_indices(v::CompoundView{DType, NDim,  SegmentNB}, ::Vararg{Colon, NDim} ) where {DType, NDim,  SegmentNB} =
    CompoundView{DType, NDim,  SegmentNB}(size(v), v.segments, v.segment_ranges, v.axis)


# Create a view of a compound view from indice:
# ---------------------------------------------

function create_view_from_indices(v::CompoundView{DType, NDim,   SegmentNB}, indices::Vararg{Any, NDim}) where {DType, NDim,  SegmentNB}

    # Slicing the compound view also slices its segments, discarding those outside the selected range.
    segments_sliced = Base.Array{AbstractArray{DType, NDim}, 1}()

    for (segment, index_range) in zip(v.segments, v.segment_ranges)
        sliced_segment = slice_segment(segment, index_range,  v.axis, indices...)
      #  println("SEGMENT: ", sliced_segment)
        sliced_segment !== nothing && push!(segments_sliced, sliced_segment)
    end
    return isempty(segments_sliced) ? nothing : concatenate(v.axis, segments_sliced...)
end

function slice_segment(segment::AbstractArray{DType, NDim}, index_range::AbstractRange{<: Integer}, axis::Int, indices::Vararg{Any, NDim}) where {DType, NDim}
    slicing_indices = Any[]

    for (i, idx) in enumerate(indices)

        # Slicing the segment uses the same indices as the compound view, except along the concat axis
        if i != axis 
            push!(slicing_indices, idx)
        else
            # If the segment is kept, adjust its concat-axis index to match its local range.
            if keepSegment(index_range, idx)
                push!(slicing_indices, adjusted_index(index_range, idx))
            else
                return nothing
            end
        end
    end
   # println("SEGMENT BEFORE: ", segment, " indices sliced: ", slicing_indices, " indices: ", indices)
    norm_indices = normalize_indices(slicing_indices)
    return segment[norm_indices...]
end

keepSegment(axis_range, Colon)              = true
keepSegment(axis_range, idx::Int)           = Base.in(idx, axis_range)
keepSegment(axis_range, idx::AbstractRange) = any(elem -> Base.in(elem, idx), axis_range)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     Simple functions:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.parent(A::CompoundView{T}) where T = A

vcat_views(arrays::Vararg{AbstractArray}) = concatenate(1, arrays...)

hcat_views(arrays::Vararg{AbstractArray}) = concatenate(2, arrays...)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get/Set element:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# 'Normal' index
@propagate_inbounds function _getElement(a::CompoundView{DType, N}, indices::NTuple{N, Any}) where {DType, N}
    index_axis = indices[a.axis]
   
    for (i, index_range) in enumerate(a.segment_ranges)
        if (Base.in(index_axis, index_range) )
            return a.segments[i][_getIndex(indices,  a.segment_ranges[i], index_axis, a.axis)...]
        end
    end

    throw(ArgumentError("The indices are out of bounds"))
end

@propagate_inbounds function _setElement!(a::CompoundView{DType, N}, val::DType, indices::NTuple{N, Any}) where {DType, N}
    index_axis = indices[a.axis]
   
    for (i, index_range) in enumerate(a.segment_ranges)
        if (Base.in(index_axis, index_range) )
            a.segments[i][_getIndex(indices,  a.segment_ranges[i], index_axis, a.axis)...] = val
        end
    end

    throw(ArgumentError("The indices are out of bounds"))
end


# CartesianIndex
@propagate_inbounds _getElement(a::CompoundView{DType, N}, I::CartesianIndex{N})  where {DType, N} = _getElement(a, Tuple(I))
@propagate_inbounds _setElement!(a::CompoundView{DType, N}, I::CartesianIndex{N}) where {DType, N} = _setElement!(a, Tuple(I))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #




function getindex(a::CompoundView{T, N}, i::Int) where {T, N}
    I = fromLinearToCartesian(i, a)
end

@inline function _unsafe_getindex(A::CompoundView{T,N}, indices::Vararg{Int,N}) where {T,N}
    parent_axes = axes(A)
    # Convert the index relative to the reshaped array into a linear index 'i',
    # then back to Cartesian indices 'I' in the parent to handle offsets and reshaping correctly.
    i = _offset_index(_cartesian_to_linear(size(A), indices), parent_axes)
    I = _unravel_index(parent_axes, A.mi, i)
    return @inbounds A[I...]
end



@propagate_inbounds function _getIndex(indices::NTuple{N, Any}, index_range::AbstractRange, index_axis, axis::Int) where  {N}
        index_axis_adjusted = adjusted_index(index_range, index_axis)
        return [ (i == axis) ? index_axis_adjusted : indices[i] for i in 1:N]
end




# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Reshape:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# When data becomes non-contiguous, libraries like NumPy allocate a new array.
# It feels wasteful to allocate a new array just to remove a few rows or columns,
# but reshaping a pre-allocated compound view would be overly complex and not worth the effort.

# TODO: Change to ndarray
create_view_from_shape(v::CompoundView{T, M}, shape::NTuple{N}) where {T, M, N} = reshape( copyto!(Array{T, M}(undef, size(v)), v), shape)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           isConcatenable:                                     #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# To be concatenable all the dimensions except for the concatenation dimension must be the same
function isConcatenable(ref_shape::NTuple{NDim}, shape::NTuple{NDim}, axis::Int) where NDim
    for i in 1:NDim
        (i == axis ) && continue
        ref_shape[i] != shape[i] && return false
    end
    return true
end

function areConcatenable(shapes::NTuple{SegmentNB, NTuple{NDim, Int}}, axis::Int) where {SegmentNB, NDim}
    ref_shape = first(shapes)
     for i in 1:SegmentNB
        isConcatenable(ref_shape, shapes[i], axis) || return false
    end
    return true
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           adjusted index:                                     #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


adjusted_index(axis_range, idx::Colon) = idx

adjusted_index(axis_range, idx::Integer) = idx - first(axis_range) + 1 

function adjusted_index(axis_range::AbstractRange{I}, idx::AbstractRange{I}) where {I <: Integer}
    # Clip the requested index to the segment's range
    start_idx = max(first(idx), first(axis_range))
    stop_idx  = min(last(idx), last(axis_range))
    
    # Convert to local coordinates (1-based within segment)
    start_local = start_idx - first(axis_range) + 1
    stop_local  = stop_idx - first(axis_range) + 1

    return start_local:step(idx):stop_local
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     normalized indices index:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function normalize_indices(indices)
    map(i -> isa(i, Integer) ? UnitRange(i, i) : i, indices)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                    Iteration function:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #
Base.IndexStyle(::Type{CompoundView}) = IndexCartesian()


# function Base.iterate(a::CompoundView{DType,N}) where {DType,N}
#     isempty(a) && return nothing
#     first_index = CartesianIndex(_getIndex(a))  # (1, 1, ..., 1)
#     return (a[first_index], first_index)
# end

# function Base.iterate(a::CompoundView{DType,N}, state::Base.AbstractCartesianIndex{N}) where {DType,N}
#     next_state = inc(state, size(a))  # move to next Cartesian index
#     next_state === nothing && return nothing
#     return (a[next_state], next_state)
# end

# TO DO if compound view has only one element then it is a view of another compound view

