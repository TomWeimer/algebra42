include("other/constant.jl")
include("container/NDArray/AbstractNDArray.jl")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           Broadcast:                                          #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type AbstractBroadcasted42 end

# ======== Types used ================================================== #

# We restrain the types that can be used in the broadcasting to make things simpler

# Types that can be used as operator are:
struct BroadcastedFunction42{f <: Function} end
const Operator = Union{+, -, *, /, BroadcastedFunction42}


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       Broadcasted42:                                          #
#_______________________________________________________________________________________________#

# lazy container
struct Broadcasted42{Style, Shape, F, Args<:Tuple}
    f::F
    args::Args
end

const Broadcasted42Fast = Broadcasted42{<:Any,<:Tuple{Any}}

# ======== Constructors ================================================== #

Broadcasted42(f::F, args...) where {F <: Operator} = Broadcasted42(f, args)

# 1. Outer Constructor (Function Barrier)
Broadcasted42(f::F, args::Tuple) where {F <: Operator} =  (
    # Get the combined style trait (this should be type-stable already)
    Style = combine_styles(args...); 
    
    # Calculate the shape as a runtime Tuple (e.g., (10, 5))
    computed_shape = obtain_broadcast_shape(args); 
    
    # Call the specialized implementation helper, passing the shape as a Val argument.
    _Broadcasted42(f, args, Style, Val(computed_shape))
)

# 2. Specialized Implementation (Lifts Shape Value to Type Parameter)
function _Broadcasted42(
    f::F, 
    args::Tuple, 
    ::Style, 
    ::Val{Shape}
) where {F, Style, Shape}
    
    # Now, Shape is a compile-time constant (e.g., (10, 5) which is a Tuple value)
    # Include the Shape as a new type parameter for the struct.
    return Broadcasted42{Style, F, Shape}(f, args)
end

# ======== Axes ================================================== #

Base.axes(::Broadcasted42{Style, F, Shape}) where {Style, F, Shape} = map(Base.OneTo, Shape)

# ======== IndexStyle ================================================== #

# Entry function
Base.IndexStyle(bc::Broadcasted42) = IndexStyle(typeof(bc))

# The broadcasted shape is a single vector
Base.IndexStyle(::Type{<:Broadcasted42Fast}) = IndexLinear()

# The broadcasted shape is a nd array
Base.IndexStyle(::Type{<:Broadcasted42{<:Any}}) = IndexCartesian()

# ======== LinearIndices ================================================== #

Base.LinearIndices(bc::Broadcasted42Fast) = LinearIndices(axes(bc))::LinearIndices{1}

# ======== Iteration ================================================== #

Base.eachindex(bc::Broadcasted42) = _eachindex(axes(bc))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices42(t)

function Base.iterate(bc::Broadcasted42)
    iter = eachindex(bc)
    iterate(bc, (iter,))
end
Base.@propagate_inbounds function Base.iterate(bc::Broadcasted42, state)
    y = iterate(state...)
    y === nothing && return nothing
    i, newstate = y
    return (bc[i], (state[1], newstate))
end

# ======== ndims ================================================== #


Base.ndims(bc::Broadcasted) = ndims(typeof(bc))
Base.ndims(::Type{<:Broadcasted{<:Any,<:NTuple{N,Any}}}) where {N} = N

Base.size(bc::Broadcasted) = map(length, axes(bc))
Base.length(bc::Broadcasted) = prod(size(bc))


# ======== getindex ================================================== #

@inline function Base.getindex(bc::Broadcasted42, I::Union{ Integer, AllCartesianIndex })
    @boundscheck checkbounds(bc, I)
    return _getBroadcastValue()
    
end

# For Broadcasted
Base.@propagate_inbounds function _getBroadcastValue(bc::Broadcasted42, I)
    args = _getBroadcastArgsValue(bc.args, I)
    return bc.f(args...)
end

# scalar
Base.@propagate_inbounds _getBroadcastArgsValue(A::Union{AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices

# arrays
Base.@propagate_inbounds _getBroadcastArgsValue(A, I) = A[I]


@inline Base.checkbounds(bc::Broadcasted42,  I::Union{ Integer, AllCartesianIndex }) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       BroadcastedStyle42:                                     #
#_______________________________________________________________________________________________#

# determines the type of the output

abstract type BroadcastStyle42 end

struct Style{T} <: BroadcastStyle end

# styles for scalar
struct ScalarStyle <: BroadcastStyle42 end

abstract type AbstractArrayStyle42{N} <: BroadcastStyle end

# styles for NDArray
struct NDArrayStyle{N} <: AbstractArrayStyle42{N} end

# styles for Array
struct ArrayStyle42{N} <: AbstractArrayStyle42{N} end


# One type
broadcast_style(::Type{<:Number}) = ScalarStyle()
broadcast_style(::Type{<:AbstractNDArray}) = NDArrayStyle()
broadcast_style(::Type{<:AbstractArray}) = ArratStyle42()

# two types
broadcast_style(s1::S, s2::S) where S <: BroadcastStyle42 = S()
broadcast_style(s1::S, s2::U) where {S <: BroadcastStyle42, U <: BroadcastStyle42 }  = S() # first type has priority

# ======== Obtain the broadcast style from arguments ================================================== #

# If there is no argument return scalar style
combine_styles() = ScalarStyle()

# If the argument is a broadcasted object returns its style
combine_styles(bc::Broadcasted42) = bc.style

# Other wise return the result style of the arguments
combine_styles(c)             = broadcast_style(typeof(c))
combine_styles(c1, c2)        = broadcast_style(combine_styles(c1), combine_styles(c2))
combine_styles(c1, c2, cs...) = broadcast_style(combine_styles(c1), combine_styles(c2, cs...))

# ======== Broadcast eltype ================================================== #

broadcast_eltype(bc::Broadcasted42) = Base.combine_eltypes(bc.f, bc.args)
broadcast_eltype(A) = Base.eltype(A)  # Tuple, Array, etc.

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                     Materialize: function:                                    #
#_______________________________________________________________________________________________#

# materialization
function materialize42(bc::Broadcasted42{Style, Shape}) where {Style, Shape}
    ElType = broadcast_eltype(bc)
    dest = similar(bc, ElType, Shape)
    return copy!(dest, bc)
end

# In place materialization
function materialize42!(dest, bc::Broadcasted42)
    return copy!(dest, bc)
end

# ======== similar =========================================================================== #

## Allocating the output container
Base.similar(bc::Broadcasted42, ::Type{T}) where {T} = similar(bc, T, axes(bc))

Base.similar(::Broadcasted42{ArrayStyle42{N}}, ::Type{ElType}, dims) where {N,ElType} = 
    similar(Array{ElType}, dims)

Base.similar(::Broadcasted42{NDArrayStyle{N}}, ::Type{ElType}, dims) where {N,ElType} = 
    NDArray{ElType}(dims)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                     Broadcast: function                                       #
#_______________________________________________________________________________________________#

broadcast(f::Tf, As...) where {Tf} = materialize42(broadcasted(f, As...))


# 3. The index-level value getter getindex

# 4. Special assignment/acumulation handling



# Steps with macro:
# 1. parsing
# 2. Trigger
# 3. Fusion
# 4. Execution

# broadcast style
#struct NDArrayStyle <: Base.BroadcastStyle end

# when to use the broadcast style
Base.Broadcast.BroadcastStyle(::Type{<:AbstractNDArray}) = NDArrayStyle()
Base.BroadcastStyle(::NDArrayStyle, ::NDArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::NDArrayStyle, ::Base.Broadcast.DefaultArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle, ::NDArrayStyle) = NDArrayStyle()

# This defines how Julia should allocate the output array when performing a broadcast operation involving AbstractNDArray type.
Base.similar(bc::Base.Broadcast.Broadcasted{NDArrayStyle}, ::Type{ElType}) where {ElType} = begin
    axes_bc = Base.Broadcast.axes(bc)
    dims = tuple(length.(axes_bc)...)

    dest = NDArray{ElType}(dims)
    return dest
end


# How to actually fill it with the broadcasted values
function Base.Broadcast.copy!(dest::AbstractNDArray, bc::Broadcast.Broadcasted{NDArrayStyle})
    for (idx, val) in zip(eachindex(dest), bc)
        dest[idx] = val
    end
    return dest
end


function broadcast_eltype(bc::Broadcasted42{op, T, S}) where {op, T, S}
    # Recursively find the element types of the arguments (or the types themselves if scalar)
    ElTypeL = broadcast_eltype(bc.lhs)  # e.g., Int64
    ElTypeR = broadcast_eltype(bc.rhs)  # e.g., Float64

    # Use inference to find the result type of the operation on those types
    return_type = Base._return_type(op, Tuple{ElTypeL, ElTypeR})
    
    return return_type # e.g., Float64 (because Int64 + Float64 -> Float64)

    


"""
    _broadcast_getindex(A, I)

Index into `A` with `I`, collapsing broadcasted indices to their singleton indices as appropriate.
"""
Base.@propagate_inbounds _broadcast_getindex(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds _broadcast_getindex(::Ref{Type{T}}, I) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds _broadcast_getindex(A::Tuple{Any}, I) = A[1]
Base.@propagate_inbounds _broadcast_getindex(A::Tuple, I) = A[I[1]]
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
Base.@propagate_inbounds _broadcast_getindex(A, I) = A[newindex(A, I)]