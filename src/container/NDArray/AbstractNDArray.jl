using Base: @propagate_inbounds

using .Errors: ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY

abstract type AbstractNDArray{T, N} <: AbstractArray{T, N} end

const ScalarNDArray{DType} = AbstractNDArray{DType,0}

const AllCartesianIndex{N} = Base.AbstractCartesianIndex{N}

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Functions to Overload:                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# To create a class that behave like a AbstractNDArray, you only need to overload the following functions:

# Return the indexStyle used by the array
Base.IndexStyle(::Type{<:AbstractNDArray}) = @mustoverload

# Return the indexStyle used by the array
IsContingous(::Type{<:AbstractNDArray}) = @mustoverload

# Return the shape of the array
Base.size(a::AbstractNDArray) = map(length, axes(a))

# Return the shape of the array
Base.axes(a::AbstractNDArray) = @mustoverload

# Return the parent array, or self if no parent exists
Base.parent(a::AbstractNDArray) = @mustoverload

# Create a view of the element from indices
Base.view(a::AbstractNDArray{DType,N}, inds...) where {DType,N} = @mustoverload


# Get and set elements:
# ---------------------

# Scalar
_getElement(a::ScalarNDArray) = @mustoverload
_setElement!(a::ScalarNDArray{T}, val::T) where T = @mustoverload

# LinearIndices
_getElement(a::AbstractNDArray{T, N}, i::Int) where {T, N} = @mustoverload
_setElement!(a::AbstractNDArray{T, N}, val::T, i::Int) where {T,N} = @mustoverload

# CartesianIndex
_getElement(a::AbstractNDArray{DType, N}, I::Base.AbstractCartesianIndex{N}) where {DType, N} = @mustoverload
_setElement!(a::AbstractNDArray{DType, N}, val::DType, I::Base.AbstractCartesianIndex{N}) where {DType, N} = @mustoverload

# 'Normal' indices
_getElement(a::AbstractNDArray{T, N}, indices::NTuple{N, Any})          where {T, N} = @mustoverload
_setElement!(a::AbstractNDArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = @mustoverload

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Arrray related functions:                                   #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Return the total number of elements of the array
Base.length(a::AbstractNDArray{T, 0}) where T = 1

# Return the total number of elements of the array
Base.length(a::AbstractNDArray) = prod(size(a))

# Return the type of the elements of the array
Base.eltype(::AbstractNDArray{T}) where {T} = T

# Return the number of dimension of the array
Base.ndims(::AbstractNDArray{T, N}) where {T, N} = N

Base.strides(::AbstractNDArray{T, 0}) where T = ()


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                            Implementation of AbstractArray:                                   #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Simple implementation of axes
Base.axes(a::AbstractNDArray) = map(Base.OneTo, size(a))

# Is needed when the array can be in an uninitialized state
#Base.isassigned(a::AbstractNDArray, i::Int) = checkbounds(Bool, a, i)

# Return a copy of the array
Base.copy(a::AbstractNDArray) = copy!(similar(a), a)

function Base.copy!(dest::AbstractNDArray, src::AbstractNDArray)
    @boundscheck size(dest) == size(src)
    println("want to copy shape dest: ", size(dest), " shape src: ", size(src), " indices: ", eachindex(src))
    for idx in eachindex(src)
        dest[idx] = src[idx]
        println("dest[$idx] ($(dest[idx])) = src[$idx] ($(src[idx])) ")
    end
    return dest
end

function Base.copy!(dest::AbstractNDArray{T, 0}, src::AbstractNDArray) where T
    @boundscheck size(dest) == size(src)
    data = src[()]
    dest[()] = data
    return dest
end


# Base.eachindex(A::AbstractNDArray) = eachindex(IndexStyle(A), A)

Base.eachindex(::IndexLinear, A::AbstractNDArray{T, 0}) where T = 1:1
Base.eachindex(::IndexLinear, A::AbstractNDArray{T, 1}) where T = isempty(A) ? (1:0) : axes(A)[1]
Base.eachindex(::IndexLinear, A::AbstractNDArray{T, N}) where {T, N} = isempty(A) ? (1:0) : (1:length(A))

Base.eachindex(::IndexCartesian, A::AbstractNDArray) = CartesianIndices_42(A)

# Handle the iteration of the arrays depending on their index style
Base.iterate(a::AbstractNDArray) =  _iterate_by_style((IndexStyle(a)), a)


# Return the first and last element
Base.first(a::AbstractNDArray) = a[_first_index(a)]
Base.last(a::AbstractNDArray)  = a[ _last_index(a)]


# Functions if their IndexStyle is IndexLinear :
_last_index(a::AbstractNDArray{T, 0}) where {T, N} = 1
_last_index(a::AbstractNDArray) = last(eachindex(a))

Base.lastindex(a::AbstractNDArray{T, N}) where {T, N} = last(eachindex(a))


_iterate_by_style(::IndexLinear, a::AbstractNDArray) = isempty(a) ? nothing : (a[1], 2)
Base.iterate(a::AbstractNDArray, state::Int) = state > length(a) ? nothing : (a[state], state + 1)


# Functions if their IndexStyle is IndexCartesian :

_first_index(a::AbstractNDArray{T, 0}) where {T, N} = 1
_first_index(a::AbstractNDArray{T, N}) where {T, N} = first(eachindex(a))

Base.firstindex(a::AbstractNDArray{T, N}) where {T, N} = first(eachindex(a))

_iterate_by_style(idx_style::IndexCartesian, a::AbstractNDArray{T, N}) where {T, N} = (
    firstindex = _first_index(a);
    isempty(a) ? nothing : (a[firstindex], firstindex)
)

Base.iterate(a::AbstractNDArray, state::Base.AbstractCartesianIndex) = (
    next_index = inc(state, size(a));
    next_index === nothing ? nothing : (a[next_index], next_index)
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Setindex:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.setindex!(a::ScalarNDArray{T}, val::T) where {T} = _setElement!(a, val)
Base.setindex!(a::ScalarNDArray{T}, val::T, ::Tuple{}) where {T} = _setElement!(a, val)


Base.setindex!(a::AbstractNDArray{T,N}, val::T,  i::Int) where {T, N} = (
    @boundscheck checkbounds(a,  i);
    @inbounds _setElement!(a, val, i)
)

Base.setindex!(a::AbstractNDArray{T,N}, val::T, I::Base.AbstractCartesianIndex{N}) where {T,N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _setElement!(a, val, I)
)

Base.setindex!(a::AbstractNDArray{DType, N}, val::DType, inds::Vararg{Int, N}) where {DType, N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _setElement!(a, val, inds)
)



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Getindex:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.getindex(a::ScalarNDArray)            = _getElement(a)
Base.getindex(a::ScalarNDArray, ::Tuple{}) = _getElement(a)

Base.getindex(a::ScalarNDArray, i::Int)    = (
    @boundscheck checklinear(a,  i);
    @inbounds _getElement(a, 1)
)


# Handle Linear index
Base.getindex(a::AbstractNDArray{T, N}, i::Int) where {T, N} = (
    @boundscheck checklinear(a,  i);
    @inbounds _getElement(a, i)
)

# Handle Cartesian index
Base.getindex(a::AbstractNDArray{T, N}, I::AllCartesianIndex{N}) where {T, N} = (
    @boundscheck checkbounds(a,  I);
    @inbounds _getElement(a, I)
)

# Handle multiple int index but not a cartesian index
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{Index, 1}) where {T,N} = (
    i = inds[1];
    @boundscheck checklinear(a, i);
    @inbounds _getElement(a, inds[1])
)

# Handle multiple int index but not a cartesian index
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{Index}) where {T,N} = (
    @boundscheck checkbounds(a,  inds...);
    @inbounds _getElement(a, inds)
)

# Handle index resulting in a view
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{ViewIndices}) where {T, N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds @view a[indices...]
)
    
# Handle indices resulting in a copy (fancy indexing)
Base.getindex(a::AbstractNDArray{T,N}, inds::Vararg{CopyIndices}) where {T,N} = (
    @boundscheck checkbounds(a, inds...);
    indices = Base.to_indices(a, inds);
    @inbounds fancy_index(a, indices)
)

function fancy_index(src_array::AbstractNDArray, indices)
    # Obtain shape
    src_shape = size(src_array)

    # Obtain only the fancy indices ( [1, 3], [ 1 4; 2 3], ... )
    fancy_indices = _collect_fancy(indices...)

    # We stetch all the shapes of the fancy indices to match a single dimension
    padded_fancy_shapes = obtain_padded_shapes(fancy_indices)

    # Obtain the shape of the output array
    fancy_offset, output_shape = obtain_output_shape_fancy(src_shape, fancy_indices, indices, padded_fancy_shapes)

    # Create the output array
    out = NDArray{eltype(src_array)}(output_shape)

    # Fill the output array
    _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes, fancy_offset)

    return out
end


function _fill_fancy!(out, src_array, indices, output_shape, padded_fancy_shapes::AbstractArray{<:PaddedShape}, fancy_offset)
    # We iterate over the output indices, because the reverse is not one to one 
    output_indices = CartesianIndices_42(output_shape)

    for output_idx in output_indices
        # We compute the source index with the information given by the output_idx
        src_idx = _get_src_idx(indices, output_shape, output_idx, padded_fancy_shapes, fancy_offset)
        out[output_idx] = src_array[src_idx...]
    end
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                    Get and Set index errors:                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


function Base.getindex(a::ScalarNDArray,   i::Int)
    i == 1 && return _getElement(a)
    @throw_index_error i ERR_INDEX_INT_ON_SCALAR_ARRAY
end


function Base.setindex!(a::ScalarNDArray{T}, val::T, i::Int) where T
    i == 1 && return _setElement!(a, val)
    @throw_index_error i ERR_INDEX_INT_ON_SCALAR_ARRAY
end

Base.getindex(::AbstractNDArray, inds::Tuple{})        = @throw_index_error inds ERR_EMPTY_TUPLE_ON_ND_ARRAY
#Base.getindex(::ScalarNDArray,   inds::ViewIndices...) = @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY
Base.getindex(::AbstractNDArray{T, 1},  inds::NTuple{1}) where {T} = @throw_index_error inds "Error invalid index types for 1 dimensional array"




Base.setindex!(::AbstractNDArray{T,N}, ::T, inds::Tuple{})     where {T,N} = @throw_index_error inds ERR_EMPTY_TUPLE_ON_ND_ARRAY
Base.setindex!(::ScalarNDArray{T}, ::T, inds::ViewIndices...)  where T     = @throw_index_error inds ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY
Base.setindex!(::AbstractNDArray{T, 1},  inds::NTuple{1}) where {T} = @throw_index_error inds "Error invalid index types for 1 dimensional array"



function offset(a::AbstractNDArray{T, N}, indices::Union{Base.AbstractCartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing; default_offset::Int = 0) where {T, N}
    parent_strides = strides(parent(a))
    return _offset(parent_strides, indices; default_offset = default_offset)
end


maybe_reshape_parent(A::AbstractArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
maybe_reshape_parent(A::AbstractArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
maybe_reshape_parent(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
maybe_reshape_parent(A::AbstractArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))


function printNDArray(io::IO, A::AbstractNDArray)
    dims = size(A)
    typename = nameof(typeof(A))
    
    # Header line
    println(io, "==================== $(typename) $(dims) ====================")
    println(io, "content:")

    nd = ndims(A)
    if nd == 1
        # 1D vector display
        print(io, "[ ")
        print(io, join(A, " "))
        println(io, "]")

    elseif nd == 2
        # 2D matrix display
        print(io, "[ ")
        for i in 1:size(A, 1)
            for j in 1:size(A, 2)
                print(io, A[i, j])
                if j < size(A, 2)
                    print(io, " ")
                end
            end
            if i < size(A, 1)
                println(io)  # new row
                print(io, "  ")
            else
                println(io, "]")
            end
        end

    else
        # Higher-dimensional array — print slices
        for k in 1:size(A, 3)
            println(io, "slice [:, :, $k] =")
            Base.show(io, view(A, :, :, k))
            println(io)
        end
    end

    println(io, "=========================================================")
end

function checklinear(A::AbstractArray, i::Integer)
    if i < 1 || i > length(A)
        throw(BoundsError(A, i))
    end
    return true
end

