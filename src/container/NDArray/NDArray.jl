import Base: *, +, -

struct NestedArrayIndices
    dims::NTuple{N,Int} where N
end


function Base.iterate(R::NestedArrayIndices, state=ntuple(_ -> 1, length(R.dims)))
    inds = state
    N = length(R.dims) # 3

    # stop condition
    if inds === nothing # (1, 1, 1)
        return nothing
    end

    # prepare next state
    next = collect(inds) # [1, 1, 1]

    if next[end-1] < R.dims[end-1]
        next[end-1] += 1
        return (CartesianIndex(inds), Tuple(next)) 
    elseif (next[end] < R.dims[end])
        next[end-1] = 1
        next[end] += 1
        return (CartesianIndex(inds), Tuple(next)) 
    end

    for d in N:-1:1   # rightmost first
        if next[d] < R.dims[d]
            next[d] += 1
            return (CartesianIndex(inds), Tuple(next))
        else
            next[d] = 1
        end
    end
    return (CartesianIndex(inds), nothing)
end


"""
   NDArray{DType,N}

Fields:
- `content`: Stores the array data as a 1D or 0D array.
- `shape`: Shape information for the array.
- `strides`: Tuple of strides for efficient indexing.

Provides constructors for scalars, ragged arrays, regular arrays, and matrices.
"""
mutable struct NDArray{DType,N} <: AbstractNDArray{DType,N}
   content::Union{Array{DType,1},Array{DType,0}}
   shape::Shape{N}
   strides::Tuple{Vararg{Int}}

   # Default constructor
   NDArray(content::AbstractArray{T}, shape::Shape{N}, strides) where {T,N} = new{T, N}(content, shape, strides)
end

const NestedArray{T} = AbstractArray{T, 1} # can also be single dimension array

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                      Constructors:                                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
NDArray{T}(x::Number) where {T} = (
   _init(x, T)
)

# Regular nested array
NDArray{T}(data::NestedArray{U}) where {T, U} = (
    _init(data, Shape(data), T)
)

# Ragged nested array
NDArray{Any}(data::NestedArray{T}) where {T} = (
    _init(data, Shape(data; dtype=Any), Any)
)

# From multidimensional array
NDArray{T}(data::AbstractArray{U, N}) where {T, U, N} = (
    _init(data, Shape(size(data)), T)
)

# From shape (tuple)
NDArray{T}(shape::Tuple) where {T} = (
    NDArray{T}(Shape(shape))
)

# From shape object
NDArray{T}(shape::Shape{N}) where {T, N} = (
    NDArray(Array{T}(undef, shape.length), shape, compute_strides(shape))
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                          _init:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
_init(x::Number, T::Type) = (
    _init(fill(T(x)), Shape(x))
)

# Array with explicit type and shape
_init(data::AbstractArray{U}, shape::Shape{N}, T::Type, strides=compute_strides(shape)) where {U, N} = (
    _init(_fill(data, shape, strides, T), shape, strides)
)

# Generic catch-all
_init(content::AbstractArray{T}, shape::Shape{N}, strides=compute_strides(shape)) where {T, N} = (
    println("content is now: ", content, " shape is: ", shape);
    NDArray(content, shape, strides)
)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                         _fill:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Fill using multidimentional array
function _fill(data::AbstractArray{T,N}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {T,N}
    #println("length: ", length(shape), "shape: ", shape.dims, "data shape: ", size(data), "data length: ", length(data))
    println(stderr, "from nd arrays")
    content = Array{dtype, 1}(undef, shape.length)
   for (i, val) in enumerate(data)
      println(stderr, "index: ", i, "value: $(dtype(val))")
      content[i] = dtype(val)
   end
   println("content: ", content)
   return content
end

function _fill(data::NestedArray{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    if dtype === Any
        return _fill_ragged_array(data, shape, strides, dtype)
    end
    return N ≥ 3 ? _fill_from_nested_indices(data, shape, strides, dtype) :  _fill_from_offset(data, shape, strides, dtype)
end

# Fill ragged array
function _fill_ragged_array(data::NestedArray{U}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    content = similar(data, dtype, shape.length)
    for i in eachindex(content)
        content[i] = data[i]
    end
    content
end

# Fill dim 1 and dim 2
function _fill_from_offset(data::NestedArray{U}, shape::Shape{N}, strides::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
    content = similar(data, dtype, shape.length)
    for I in CartesianIndices(size(shape))
        content[_offset(strides, Tuple(I))] = dtype(_get_nested(data, I))
    end
    content
end

# Fill dim 3+
function _fill_from_nested_indices(data::AbstractArray{U, 1}, shape::Shape{N}, ::Tuple{Vararg{Int}}, dtype::Type) where {U,N}
   content = similar(data, dtype, shape.length)

   println(stderr, "data is: ", data)
   println(stderr, "from nested indices")
   for (i, I) in enumerate( NestedArrayIndices( size(shape) ))
        println(stderr, "index: ", I, "value: $(dtype( _get_nested(data, I) ))")
      content[i] = dtype( _get_nested(data, I) )
   end
   println(stderr, "from nested arrays")

   input_indices = CartesianIndices_42(size(shape), order=RowOrder)
   column_indices = CartesianIndices_42(size(shape), order=ColOrder)

   for i in 1:length(shape)
        column_idx = column_indices[i]
        input_idx = input_indices[i]
        println(stderr, "input_idx: ", input_idx, " content_idx: ", column_idx, " value: $(dtype( _get_nested(data, input_idx) ))")
        content[i] = dtype( _get_nested(data, input_idx) )    
    end
#    for (i, I) in enumerate( CartesianIndices_42( size(shape); order=RowOrder ))
#         println(stderr, "index: ", I, "value: $(dtype( _get_nested(data, I) ))")
#       content[i] = dtype( _get_nested(data, I) )
#    end
   return content
end

function _get_nested2(data, I::AllCartesianIndex)
    tmp = data
    for i in I
        tmp = tmp[i]
    end
    return tmp
end

_get_nested(data, I::AllCartesianIndex) = foldl(getindex, Tuple(I); init=data)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Similar:                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Similar: (create array of same size but uninitialized)
Base.similar(A::AbstractNDArray{T}) where {T} = NDArray{T}(size(A))

Base.similar(A::AbstractNDArray, ::Type{T}) where {T} = NDArray{T}(size(A))

Base.similar(::AbstractNDArray{T}, dims::Tuple{Vararg{Int}}) where {T} = NDArray{T}(dims)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Other functions :                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.strides(a::NDArray{T, N}) where {T, N}     = a.strides

Base.stride(a::NDArray, k::Integer)     = ( @boundscheck checkindex(Bool, axes(a, k), k); @inbounds a.strides[k])

# Parent:
flatten(array::NDArray{DType,N}) where {DType,N} = array.content

Base.pointer(A::NDArray{T}) where T = pointer(A.content)

# # Create a ndarray from range
# function Base.reshape(range::AbstractRange{T}, shape::NTuple{N})  where {T, N}
#     length(range) == prod(shape) || throw(ArgumentError("Can't resize the range with the shape given"))
#     ndarray = NDArray{T}(shape)
#     for (i, val) in enumerate(range)
#       ndarray[i] = val 
#    end
#    return ndarray
# end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Implementation AbstractNDArray:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.IndexStyle(::Type{<:NDArray}) = IndexLinear()

Base.size(a::NDArray)  = a.shape.dims

Base.axes(a::NDArray) = map(Base.OneTo, a.shape.dims)

Base.parent(a::NDArray) = a

# function Base.view(a::NDArray, inds...)
#     println("We want A[$inds] where A is: ")
#     println(a)
#     println("-------------- Creation View (indices: $inds) ---------------- ")
#     # check bounds
#     J = to_indices(a, inds)

#     @boundscheck checkbounds(a, J...)
    
#     # drop dimension
#     J_2 =  drop_singleton_dimension(J, ndims(a))
    
#     # resize parent if needed
#     reshaped_parent = maybe_reshape_parent(a, Base.index_ndims(J_2...))
#     size_before, size_now = size(a), size(reshaped_parent)
#     println("J: $J")
#     println("J': $J_2")

#     # print the content
#     println("content: ", ((size_before != size_now) ? " (reshaped from $size_before to $size_now) " : " " ) * string(reshaped_parent) )

#     println("params: reshaped_parent: ndimsA: $(ndims(a)) $reshaped_parent, index_ndims: $(Base.index_ndims(J_2...))")
#     # create the view
#     V =  create_view(reshaped_parent, J_2...)

#     println("firstindex: ", firstindex(V))
#     println("lastindex: ", lastindex(V))
#     println("offset1: ", V.offset1)
#     println("axes: ", axes(V))
#     println("indexStyle: ", IndexStyle(V))
#      println("------------------------------------------------------------------------- \n\n")
#     return V
# end

# Get and set elements:
# ---------------------

# Scalar:
# -------

@propagate_inbounds _getElement(a::NDArray{T, 0}) where {T} = (
   a.content[1]
)

@propagate_inbounds _setElement!(a::NDArray{T, 0}, val::T) where {T} = (
   a.content[1] = val
)

# Linear index:
# ------------

@propagate_inbounds _getElement(a::NDArray{T, N}, i::Int) where {T, N} = (
   a.content[i]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, i::Int) where {T, N} = (
   a.content[i] = val
)

# Cartesian index:
# ---------------

@propagate_inbounds _getElement(a::NDArray{T, N}, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    println("index obtained: ", I, " index returned: ", (offset(a, I)));
    a.content[ offset(a, I) ]
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, I::Base.AbstractCartesianIndex{N}) where {T, N} = (
    println("index obtained: ", I, " index returned: ", (offset(a, I)));
    a.content[ offset(a, I) ] = val
)

# 'Normal' index:
# ---------------

@propagate_inbounds _getElement(a::NDArray{T, N}, indices::NTuple{N, Any}) where {T, N} = ( 
   a.content[ offset(a, indices) ] 
)

@propagate_inbounds _setElement!(a::NDArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
   a.content[ offset(a, indices) ] = val
)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Math operator Overload :                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# The content of this file permits to use the broadcast syntax suxh as v .+= λ .* v

# We create a new broadcast style and we assign it to our type
struct NDArrayStyle <: Base.BroadcastStyle end

Base.Broadcast.BroadcastStyle(::Type{<:NDArray}) = NDArrayStyle()

Base.similar(bc::Base.Broadcast.Broadcasted{NDArrayStyle}, ::Type{ElType}) where {ElType} = begin
   # T = Base.Broadcast.broadcasted_eltype(bc)
    axes_bc = Base.Broadcast.axes(bc)
    dims = tuple(length.(axes_bc)...) 

    dest = NDArray{ElType}(dims)
    return dest
end

# Copy data from a Broadcasted object to a destination MyVector
function Base.Broadcast.copy!(dest::NDArray, bc::Broadcast.Broadcasted{NDArrayStyle})
    for (i, val) in enumerate(bc)
        dest[i] = val
    end
    return dest
end


Base.BroadcastStyle(::NDArrayStyle, ::NDArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::NDArrayStyle, ::Base.Broadcast.DefaultArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle, ::NDArrayStyle) = NDArrayStyle()


# --- Addition ---
+(a::NDArray, b::NDArray) = add(a, b)
+(a::Number, b::NDArray) = +(b, a)
+(a::NDArray, b::Number) = add(a, b)

# --- Subtraction ---
-(a::NDArray, b::NDArray) = sub(a, b)
-(a::Number, b::NDArray) = -(b, a)
-(a::NDArray, b::Number) = sub(a, b)

# --- Multiplication scalaire ---
*(a::NDArray, b::Number) = prod(a, b)
*(a::Number, b::NDArray) =  *(b, a)



const Vector{DType} = NDArray{DType, 1}

# scalar 
Vector{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Vector{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Vector{DType}(data::AbstractArray{T, 1}) where {DType, T} = _init(data, Shape(size(data)), T)

# Uninitialized constructors from shape:

Vector{DType}(shape::Tuple{Int}) where {DType} = NDArray{DType}(Shape(shape))

Vector{DType}(shape::Shape{1}) where {DType} = NDArray( Array{DType}(undef, shape.length), shape, compute_strides(shape) )

function zeroVector(shape::Tuple, type::Type{T}) where {T}
    rtn = Vector{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end

const Matrix{DType} = NDArray{DType, 2}

# scalar 
Matrix{T}(scalar::Number) where {T} = _init(scalar, T)

# array
Matrix{T}(data::Collection) where {T} = _init(data, Shape(data), T)

# matrix
Matrix{DType}(data::AbstractArray{T, 2}) where {DType, T} = _init(data, Shape(size(data)), DType)

# Uninitialized constructors from shape:
Matrix{DType}(shape::Tuple) where {DType} = NDArray{DType}(Shape(shape))

Matrix{DType}(shape::Shape{2}) where {DType} = NDArray( Array{DType, 2}(undef, shape.length), shape, compute_strides(shape) )

function zeroMatrix(shape::Tuple, type::Type{T}) where {T}
    rtn = Matrix{type}(shape)
    fill!(rtn, zero(type))
    return rtn
end


function Base.hcat(A::Matrix{T}, B::Matrix{T}) where {T}
    
    size(A, 1) == size(B, 1) || throw(ArgumentError("Those matrix have not the same height"))

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


function isSquare(A::Matrix)
    return size(A, 1) == size(A, 2)
end

function identityMatrix(n, type::Type{T}) where T
    identity = Matrix{type}((n, n))

    fill!(identity, 0)

    for k in 1:n
        identity[k, k] = one(T)
    end
    return identity
end


function offset(a::NDArray{T, N}, indices::Union{Base.AbstractCartesianIndex{N}, NTuple{N, Int}, Nothing} = nothing; default_offset::Int = 0) where {T, N}
    parent_strides = strides(parent(a))
    return _offset(parent_strides, indices; default_offset = default_offset)
end

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