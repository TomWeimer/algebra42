struct CompoundView{DType, NDim, ParentNB} <: AbstractNDArray{DType, NDim}
    shape::NTuple{NDim}
    parents::NTuple{ParentNB, AbstractArray{DType, NDim}}
    offsets::NTuple{ParentNB, Int}
    axis::Int
end


function isConcatenable(ref_shape::NTuple{NDim}, shape::NTuple{NDim}, axis::Int) where NDim
    for i in 1:NDim
        (i == axis ) && continue
        ref_shape[i] != shape[i] && return false
    end
    return true
end


function areConcatenable(shapes::NTuple{ParentNB, NTuple{NDim, Int}}, axis::Int) where {ParentNB, NDim}
    ref_shape = first(shapes)

     for i in 1:NDim
        isConcatenable(ref_shape, shapes[i], axis) || return false
    end
    return true
end

function obtainConcatShape(shapes::NTuple{ParentNB, NTuple{NDim, Int}}, shapes_along_axis::AbstractArray{Int, 1}, axis::Int) where {ParentNB, NDim}
    ref_shape = first(shapes)
    new_dim =  sum(shapes_along_axis)
    return Tuple((i == axis) ? new_dim : ref_shape[i]  for i in 1:NDim )
end

function obtainOffsets(shapes_along_axis::AbstractArray{Int, 1})
    offset = 1
    offsets = Int[]
    N = length(shapes_along_axis)

    for i in 1:(N - 1)
        offset += shapes_along_axis[i]
        push!(offsets, offset)
    end
    pushfirst!(offsets, 1)
    return offsets
end


function concatenate(axis::Int, parents::Vararg{AbstractNDArray{DType, NDim}, ParentNB}) where {DType, NDim, ParentNB}

    shapes = size.(parents)

    areConcatenable(shapes, axis) || throw(ArgumentError("Those arrays have not the same size along $axis"))

    shapes_along_axis = getindex.(shapes, axis)

    shape = obtainConcatShape(shapes, shapes_along_axis, axis)

    offsets = obtainOffsets(shapes_along_axis)

    return CompoundView{DType, NDim, ParentNB}(shape, parent, offsets, axis)
end


# TO DO if compound view has only one element then it is a view of another compound view

function create_view_from_indices(parent::CompoundView{DType, NDim, ParentNB}, ::Vararg{Colon, NDim}) where {DType, NDim, ParentNB}
    return CompoundView{DType, NDim, 1}(size(parent), (parent, ), parent.offsets, parent.axis)
end

function create_view_from_indices(parent::CompoundView{DType, NDim, ParentNB}, indices::Vararg{Any, NDim}) where {DType, NDim, ParentNB}
    parents = obtainParentsFromIndices(parent, indices)
    return concatenate(parent.axis, parents...)
end

function obtainParentsFromIndices(parent::CompoundView{DType, NDim, ParentNB}, indices::Vararg{Any, NDim}, axis::Int) where {DType, NDim, ParentNB}
    old_parents = parent.parents

    new_parents = AbstractArray{DType, NDim}[]

    for (i, old_parent) in enumerate(old_parents)
        start = parent.offsets[i]
        stop = ( i == ParentNB) ? parent.shape[ParentNB] : parent.offsets[i] - 1
        to_keep, new_parent = updateParentFromIndices(old_parent, start:stop, parent.offsets[i], indices, axis)
        to_keep && push!(new_parents, new_parent)
    end

    return new_parents
end

function updateParentFromIndices(old_parent::AbstractArray{DType, NDim}, axis_range::AbstractRange, indices::Vararg{Any, NDim}, axis::Int) where {DType, NDim}
    indices_parent = []

    for (i, idx) in enumerate(indices)
        if i != axis 
            push!(indices_parent, idx)
        else
            to_keep, new_idx = obtainModifiedIndex(axis_range, idx)
            !to_keep && return false, nothing
            push!(indices_parent, new_idx)
        end
    end
    return true, old_parent[indices_parent...]
end

function obtainModifiedIndex(axis_range, idx)
    to_keep = keepParentOrNot(axis_range, idx)
    !to_keep && return false, nothing
    return modifiedIndex(axis_range, idx)
end

keepParentOrNot(axis_range, Colon) = true

function keepParentOrNot(axis_range, idx::Int)
    return Base.in(idx, axis_range)
end

function keepParentOrNot(axis_range, idx::AbstractRange)
    return any(elem -> Base.in(elem, idx), axis_range)
end

modifiedIndex(axis_range, idx::Union{Colon, Integer}) = idx

function modifiedIndex(axis_range, idx::AbstractRange)
    smallest_stop = last(axis_range) < last(idx) ? last(axis_range) : last(idx)
    biggest_start = first(axis_range) < first(idx) ? first(idx) : first(axis_range)
    return biggest_start:step(idx):smallest_stop
end



function Base.hcat(A::Matrix{T}, B::Matrix{T}) where {T}
    
    areConcatenable || throw(ArgumentError("Those matrix have not the same height"))

    shape = ( size(A, 1), size(A, 2) + size(B, 2) )

    dest = Matrix{T}(shape)

    m, n1 = size(A, 1), size(A, 2)
    
    for j in 1:shape[2]
        
        if (j <= n1)
            for i in 1:m
                dest[i, j] = A[i, j]
            end
        else
            for i in 1:m
                dest[i, j] = B[i, j - n1]
            end
        end
    end
    return dest
end

function Base.vcat(A::Matrix{T}, B::Matrix{T}) where {T}
    
    size(A, 2) == size(B, 2) || throw(ArgumentError("Those matrix have not the same height"))

    shape = ( size(A, 1) + size(B, 1), size(A, 2) )

    dest = Matrix{T}(shape)

    m1 = size(A, 1)
    
    for j in 1:shape[2]
        for i in 1:shape[1]
            dest[i, j] = (i <= m1) ? A[i, j] : B[i - m1, j]
        end
    end
    return dest
end


Base.size(array::CompoundView) = array.shape

Base.ndims(array::CompoundView{T, N}) where {T, N} = N

Base.eltype(A::CompoundView{T}) where {T} = T


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Constructors:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


function copy(toCopy::CompoundView{DType, NDim}; ndarray = true) where {DType, NDim}
    shape = size(toCopy)
    copy = (ndarray) ? NDArray{DType}(shape) : Array{DType, NDim}(undef, shape...)
    
    for I in CartesianIndices(shape)
        copy[I] = toCopy[I]
    end

    return copy
end

# For this case if the compound view is reshaped we create a copy
function create_view_from_shape(parent::CompoundView{T, M}, shape::NTuple{N}) where {T, M, N}
    copied = copy(parent)
    return reshape(copy, shape)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Set Index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# From a view change the index with the offset
function Base.setindex!(v::CompoundView{DType}, val::DType, indices::Vararg{Int}) where {DType}
    parentIndex = findParentConcerned(v, indices)
    findParentConcerned = v.parents[parentIndex]
    modifiedIndex = modifiedIndex(v.offsets, parentIndex, indices[v.axis])
    _setElement!(findParentConcerned, _offset(findParentConcerned,  modifiedIndex...), val)
end

Base.setindex!(v::CompoundView{DType,N}, val::DType, I::CartesianIndex) where {DType,N} = setindex!(v, val, Tuple(I)...)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get Index:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Normal indexing
function Base.getindex(v::CompoundView{DType,N}, indices::Vararg{ElementIndex,N}) where {DType,N} # return an element
    # Check if the index are valid
    (elementsAreValid(size(v), indices...)) || throw(DomainError("Indexes out of bounds"))

   # println("v.parent: ", v.parent, " indices: ", indices, " offset: ", _offset(v, indices...))

   parentIndex = findParentConcerned(v, indices)
    findParentConcerned = v.parents[parentIndex]
    modifiedIndex = modifiedIndex(v.offsets, parentIndex, indices[v.axis])
    
    # If the index is valid return the associated elements
    return item(findParentConcerned, _offset(v, modifiedIndex...))
end

# Must return an element
Base.getindex(v::CompoundView{T, N}, I::CartesianIndex{N}) where {T, N} = Base.getindex(v, Tupel(I)...)

# Must expand indexing
function Base.getindex(v::NDArrayView{DType,N}, indices::Vararg{ElementIndex}) where {DType,N}
    # Check if the number of index is valid
    dimIndices, dimView = length(indices), ndims(v)

    ( dimIndices <=  dimView) || throw(DomainError("The number of index provided is too big"))

    # Expand with colon if needed
    indicesStretched = (dimIndices == dimView) ? indices : ntuple(i -> i <= dimIndices ? indices[i] : Colon(), dimView)

    return v[indicesStretched...]
end


# TODO handle view that becomes a scalar
# function Base.getindex(v::CompoundView{DType,0}, ::Tuple{}) where {DType}
#     return item(v.parent, _offset(v, ()))
# end

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

# #handle scalar view
# function _offset(view::CompoundView{T, 0}, ::Tuple{})::Int where{T}
#     return view.initial_offset + 1# start at initial offset
# end
