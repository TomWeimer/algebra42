# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           BroadcastStyle42:                                   #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type BroadcastStyle42 end

# ======== Broadcast styles =================================================================== #

# Default Style:

struct ScalarStyle <: BroadcastStyle42 end

broadcast_style(::Type{<:Number}) = ScalarStyle()


# Array Styles:

abstract type AbstractArrayStyle42{N} <: BroadcastStyle42 end

struct NDArrayStyle{N} <: AbstractArrayStyle42{N} end

broadcast_style(::Type{<:AbstractNDArray{N}}) where N = NDArrayStyle{N}()


# Base Array
struct ArrayStyle42{N} <: AbstractArrayStyle42{N} end

broadcast_style(::Type{<:AbstractArray{N}}) where N = ArrayStyle42{N}()

# ======== Obtain the broadcast style from two styles ========================================= #

broadcast_style(s1::ScalarStyle, s2::NDArrayStyle{N}) where N = NDArrayStyle{N}()
broadcast_style(s1::NDArrayStyle{N}, s2::ScalarStyle) where N = NDArrayStyle{N}()

broadcast_style(s1::ScalarStyle, s2::ArrayStyle42{N}) where N = ArrayStyle42{N}()
broadcast_style(s1::ArrayStyle42{N}, s2::ScalarStyle) where N = ArrayStyle42{N}()

broadcast_style(s1::S, s2::S) where S <: BroadcastStyle42 = S()
broadcast_style(s1::S, s2::U) where {S <: BroadcastStyle42, U <: BroadcastStyle42 }  = S() # first type has priority


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                           Broadcasted42:                                      #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct Broadcasted42{Style, Shape, F, Args<:Tuple} <: Base.AbstractBroadcasted
    f::F
    args::Args
end

# ======== Aliases ============================================================================ #

# The output is a vector
const BroadcastedVector = Broadcasted42{<:Any,<:Tuple{Any}}

# ======== Constructors ======================================================================= #

# We try to make the Broadcasted42 inferrable due to the use of the internal function Base.combine_eltypes

# Vararg
Broadcasted42(f::F, args...) where F <: Function = Broadcasted42(f, args)

# Tuple
function Broadcasted42(f::F, args::Tuple) where F <: Function
    println("args are: ", args)
    Style          = combine_styles(args...)
    computed_shape = obtain_broadcast_shape(args)
    _Broadcasted42(Style, Val(computed_shape), f, args)
end

_Broadcasted42(::Style, ::Val{Shape}, f::F, args::Tuple) where {Style, Shape, F} = (
    Broadcasted42{Style, Shape, Core.Typeof(f), typeof(args)}(f, args)
)

# ======== Core properties ==================================================================== #

Base.axes(::Broadcasted42{Style, Shape, F}) where {Style, Shape, F} = map(Base.OneTo, Shape)
Base.size(bc::Broadcasted42)   = map(length, axes(bc))
Base.length(bc::Broadcasted42) = prod(size(bc))

# Needed because N is in a nested type
Base.ndims(::Type{<:Broadcasted42{Style, Shape}}) where {Style, Shape} = length(Shape)
Base.ndims(bc::Broadcasted42) = ndims(typeof(bc))


# ======== IndexStyle ========================================================================= #

# Entry function ( Needed because N is in a nested type )
Base.IndexStyle(bc::Broadcasted42) = IndexStyle(typeof(bc))

# The broadcasted shape is a single vector
Base.IndexStyle(::Type{<:BroadcastedVector}) = IndexLinear()

# The broadcasted shape is a nd array
Base.IndexStyle(::Type{<:Broadcasted42{<:Any}}) = IndexCartesian()

# ======== Iteration ========================================================================== #

Base.eachindex(bc::Broadcasted42) = _eachindex(axes(bc))

# Return range
_eachindex(t::Tuple{Any}) = t[1]

# Return CartesianIndices
_eachindex(t::Tuple) = CartesianIndices(t)

# TODO:
#_eachindex(t::Tuple) = CartesianIndices42(t)

function Base.iterate(bc::Broadcasted42)
    iter = eachindex(bc)
    # Return a tuple to be inferrable, type is always tuple
    iterate(bc, (iter,))
end

Base.@propagate_inbounds function Base.iterate(bc::Broadcasted42, state)
    y = iterate(state...)
    y === nothing && return nothing
    idx, newstate = y
    return (bc[idx], (state[1], newstate))
end

# Needed because if the Broadcasted42 is a vector it will be called with a linear index
Base.LinearIndices(bc::BroadcastedVector) = LinearIndices(axes(bc))::LinearIndices{1}

# ======== Similar function =================================================================== #

## Allocating the output container
Base.similar(bc::Broadcasted42, ::Type{T}) where {T} = similar(bc, T, axes(bc))

Base.similar(::Broadcasted42{ScalarStyle}, ::Type{ElType}, dims) where {ElType} = zero(ElType)

Base.similar(::Broadcasted42{ArrayStyle42{N}}, ::Type{ElType}, dims) where {N,ElType} = 
    similar(Array{ElType}, dims)

Base.similar(::Broadcasted42{NDArrayStyle{N}}, ::Type{ElType}, dims) where {N,ElType} =  
    NDArray{ElType}(dims)


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                           getindex:                                           #
#_______________________________________________________________________________________________#

# Because Broadcasted42 is a lazy types, the values is computed only when accessed ( using getindex/bc[I] )

# Evaluate the value of the broadcast at index I
@inline function Base.getindex(bc::Broadcasted42, I::BroadcastIndex)
    @boundscheck checkbounds(bc, I)
    args = _eval_broadcast_args(bc.args, I)
    Base.Broadcast
    return bc.f(args...)
end

@inline function Base.getindex(bc::Broadcasted42{ScalarStyle})
    args = _eval_broadcast_args(bc.args, 1)
    return bc.f(args...)
end

# Evaluate the args to passed to f at index I
Base.@propagate_inbounds function _eval_broadcast_args(A::Tuple, I)
    return ntuple(j -> _eval_broadcast_args(A[j], I), length(A))
end

# Scalars just ignore I
Base.@propagate_inbounds _eval_broadcast_args(x::ScalarArgs, I) = x[]
# Arrays use the index
Base.@propagate_inbounds _eval_broadcast_args(x::AbstractArray, I) = x[I]
Base.@propagate_inbounds _eval_broadcast_args(bc::Broadcasted42, I) = bc[I]

# ======== Checkbounds ======================================================================== #

@inline Base.checkbounds(bc::Broadcasted42,  I::BroadcastIndex) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))


#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                           Materialize:                                        #
#_______________________________________________________________________________________________#

# Return the eltype of the output container:

# ======== Broadcast ElType =================================================================== #

# Base.combine_eltypes is the internal function used that where we need Broadcasted to be inferrable
# This returns the right type returned by the function f used with those args
broadcast_eltype(bc::Broadcasted42) = Base.Broadcast.combine_eltypes(bc.f, bc.args)

# For other types
broadcast_eltype(A) = Base.eltype(A)  # Tuple, Array, etc.

Base.Broadcast._broadcast_getindex_eltype(bc::Broadcasted42) = Base.Broadcast.combine_eltypes(bc.f, bc.args)

# ======== Materialize ======================================================================== #

# Creation of the output container:

# The type of the container:    is returned by the BroadcastStyle of the Broadcasted
# The eltype of the container:  is returned by the broadcast_eltype function
# The shape was computed as an inferrable val when constructing the Broadcasted object
# The content of the container is evaluated in the copy! function

function materialize42(bc::Broadcasted42{ScalarStyle})
    println("enter here: ", bc)
    return bc[]
end

function materialize42(bc::Broadcasted42{Style, Shape}) where {Style, Shape}
    ElType = broadcast_eltype(bc)
    dest   = similar(bc, ElType, Shape)
    return copy!(dest, bc)
end

materialize42!(dest, bc::Broadcasted42) = copy!(dest, bc)

materialize42!(op::Op, dest, bc::Broadcasted42) where {Op <: InplaceOp} = copy!(op, dest, bc)

function Base.copy!(dest,  bc::Broadcasted42{ScalarStyle})
    dest = bc[]
end

function Base.copy!(dest, bc::Broadcasted42)
    @boundscheck size(dest) == size(bc);
    for i in eachindex(bc)
        dest[i] = bc[i]
    end
    println("enter here2: ", dest)
    dest
end

function Base.copy!(op::Op, dest, bc::Broadcasted42) where {Op <: InplaceOp} 
    @boundscheck size(dest) == size(bc);
    for i in eachindex(bc)
        dest[i] = op(dest[i], bc[i])
    end
    println("enter here3: ", dest)
    dest
end

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                     Broadcast: function                                       #
#_______________________________________________________________________________________________#

# ======== Broadcastable ======================================================================== #

# broadcastable(x::Union{Symbol,AbstractString,Function,UndefInitializer,Nothing,RoundingMode,Missing,Val,Ptr,AbstractPattern,Pair,IO,CartesianIndex}) = Ref(x)
# broadcastable(::Type{T}) where {T} = Ref{Type{T}}(T)

#broadcastable(x::Union{AbstractArray,Number,AbstractChar,Ref,Tuple,Broadcasted42}) = x
# Default to collecting iterables — which will error for non-iterables
#broadcastable(x) = collect(x)

#@inline broadcasted(style::BroadcastStyle42, f::F, args...) where {F} = Broadcasted42(style, f, args)

# ======== Broadcastable ======================================================================== #


# determines the type of the output

# If the argument is a broadcasted object returns its style


# ======== Combine styles from Broadcasted42 arguments ======================================== #

# If there is no more argument return scalar style
combine_styles() = ScalarStyle()



# Other wise return the result style of the arguments
combine_styles(c::BroadcastStyle42) = c
combine_styles(bc::Broadcasted42{Style}) where Style = Style()
combine_styles(c)             = broadcast_style(typeof(c))
combine_styles(c1, c2)        = broadcast_style(combine_styles(c1), combine_styles(c2))
combine_styles(c1, c2, cs...) = broadcast_style(combine_styles(c1), combine_styles(c2, cs...))