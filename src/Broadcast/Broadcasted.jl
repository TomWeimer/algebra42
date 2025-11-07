# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                         BroadcastStyle42                                         #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

abstract type BroadcastStyle42 end

# ═══════════════════════════════════════ broadcast styles ═══════════════════════════════════════ #

abstract type AbstractArrayStyle42{N} <: BroadcastStyle42 end

broadcast_style(s1::S, s2::S) where S <: BroadcastStyle42 = S()

# By default first type has priority
broadcast_style(s1::S, s2::U) where {S <: BroadcastStyle42, U <: BroadcastStyle42 }  = S() 

# ──── scalar style  ───────────────────────────────────────────────────────────────────────────── #

struct ScalarStyle <: BroadcastStyle42 end

broadcast_style(::Type{<:Number}) = ScalarStyle()

# ──── ndarray style ───────────────────────────────────────────────────────────────────────────── #

struct NDArrayStyle{N} <: AbstractArrayStyle42{N} end

broadcast_style(::Type{<:AbstractNDArray{N}}) where N = NDArrayStyle{N}()

broadcast_style(s1::ScalarStyle, s2::NDArrayStyle{N}) where N = NDArrayStyle{N}()
broadcast_style(s1::NDArrayStyle{N}, s2::ScalarStyle) where N = NDArrayStyle{N}()


# ──── array style ─────────────────────────────────────────────────────────────────────────────── #

struct ArrayStyle42{N} <: AbstractArrayStyle42{N} end

broadcast_style(::Type{<:AbstractArray{N}}) where N = ArrayStyle42{N}()

broadcast_style(s1::ScalarStyle, s2::ArrayStyle42{N}) where N = ArrayStyle42{N}()
broadcast_style(s1::ArrayStyle42{N}, s2::ScalarStyle) where N = ArrayStyle42{N}()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                          Broadcasted42                                           #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct Broadcasted42{Style, Shape, F, Args<:Tuple} <: Base.AbstractBroadcasted
    f::F
    args::Args
end

# ──── aliases ─────────────────────────────────────────────────────────────────────────────────── #

const BroadcastedVector = Broadcasted42{<:Any,<:Tuple{Any}} # The output is a vector

# ──── constructor ─────────────────────────────────────────────────────────────────────────────── #
            
Broadcasted42(f::F, args...) where F <: Function = Broadcasted42(f, args)

function Broadcasted42(f::F, args::Tuple) where F <: Function
    Style = combine_styles(args...)
    Shape = obtain_broadcast_shape(args)
    _Broadcasted42(Style, Val(Shape), f, args)
end

_Broadcasted42(::Style, ::Val{Shape}, f::F, args::Tuple) where {Style, Shape, F} = (
    Broadcasted42{Style, Shape, Core.Typeof(f), typeof(args)}(f, args)
)

# ──── core properties ─────────────────────────────────────────────────────────────────────────── #

Base.axes(::Broadcasted42{Style, Shape, F}) where {Style, Shape, F} = map(Base.OneTo, Shape)
Base.size(bc::Broadcasted42)   = map(length, axes(bc))
Base.length(bc::Broadcasted42) = prod(size(bc))

Base.ndims(::Type{<:Broadcasted42{Style, Shape}}) where {Style, Shape} = length(Shape)
Base.ndims(bc::Broadcasted42) = ndims(typeof(bc))

# ──── similar functions ───────────────────────────────────────────────────────────────────────── #

# Allocating the output container
Base.similar(bc::Broadcasted42, ::Type{T}) where {T} = similar(bc, T, axes(bc))

Base.similar(::Broadcasted42{ScalarStyle}, ::Type{ElType}, dims) where {ElType} = zero(ElType)

Base.similar(::Broadcasted42{ArrayStyle42{N}}, ::Type{ElType}, dims) where {N,ElType} = 
    similar(Array{ElType}, dims)

Base.similar(::Broadcasted42{NDArrayStyle{N}}, ::Type{ElType}, dims) where {N,ElType} =  
    NDArray{ElType}(dims)

# ──── IndexStyle ──────────────────────────────────────────────────────────────────────────────── #

Base.IndexStyle(bc::Broadcasted42) = IndexStyle(typeof(bc))

Base.IndexStyle(::Type{<:BroadcastedVector}) = IndexLinear()

Base.IndexStyle(::Type{<:Broadcasted42{<:Any}}) = IndexCartesian()

# ──── index -─--───────────────────────────────────────────────────────────────────────────────── #

Base.eachindex(bc::Broadcasted42) = _eachindex(axes(bc))

_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices42(t)

Base.LinearIndices(bc::BroadcastedVector) = LinearIndices(axes(bc))::LinearIndices{1}

@inline Base.checkbounds(bc::Broadcasted42,  I::BroadcastIndex) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))

# ──── iteration -─--───────────────────────────────────────────────────────────────────────────── #

function Base.iterate(bc::Broadcasted42)
    iter = eachindex(bc)
    iterate(bc, (iter,))
end

@propagate_inbounds function Base.iterate(bc::Broadcasted42, state)
    y = iterate(state...)
    y === nothing && return nothing
    idx, newstate = y
    return (bc[idx], (state[1], newstate))
end


# ══════════════════════════════════ obtain output information ═══════════════════════════════════ #
            
# Before evaluating the result of the broadcasted object, we need to create a container:
# To create the right container, we need those information:

# 1. The type of the output:    returned by the BroadcastStyle of the Broadcasted
# 2. The output's shape:        computed in the constructor of the Broadcasted object
# 3. The eltype of the output:  returned by the compiler

# ──── obtain output type ──────────────────────────────────────────────────────────────────────── #

# We compute the ouput type (BroadcastStyle) when creating the Broadcasted object
# This is done by combining the styles of it's arguments:

# If there is no more argument return scalar style (default style)
combine_styles() = ScalarStyle()

combine_styles(c::BroadcastStyle42) = c
combine_styles(bc::Broadcasted42{Style}) where Style = Style()
combine_styles(c)             = broadcast_style(typeof(c))
combine_styles(c1, c2)        = broadcast_style(combine_styles(c1), combine_styles(c2))
combine_styles(c1, c2, cs...) = broadcast_style(combine_styles(c1), combine_styles(c2, cs...))

# ──── obtain broadcast shape ──────────────────────────────────────────────────────────────────── #
# The broadcast shape, not only used with Broadcasted object, follows the broadcasting rules
 
obtain_broadcast_shape(arrays::Tuple{Vararg{Number}}) = (
    return ntuple(i -> i, 0)
)

obtain_broadcast_shape(arrays::Tuple) = (
    shapes = padded_shapes(arrays);
    obtain_broadcast_shape(shapes)
)

function obtain_broadcast_shape(
    padded_shapes::NTuple{N, PaddedShape{<:Any, MaxDim}}) where {N, MaxDim}

    # We obtain the broadcast shape by taking the maximum of each dimension
    broadcast_shape = ntuple(i -> maximum(ps[i] for ps in padded_shapes), MaxDim)

    # We check that the broadcasted shape is valid
    valid = all(i->all(ps[i] == 1 || ps[i] == broadcast_shape[i] for ps in padded_shapes), 1:MaxDim)
    valid || throw( DomainError("The indices entered are not compatible for broadcasting") )
    return broadcast_shape
end

# replace indices of stretched dimensions by 1
broadcast_index(idx::AllCartesianIndex, sizeA::Tuple) = (
    ntuple(d -> sizeA[d] == 1 ? 1 : idx[d], length(sizeA))
)

# ──── obtain output eltype ────────────────────────────────────────────────────────────────────── #

# To obtain the eltype of our output before evaluating the broadcasted functions, we need
# to call an internal function asking the compiler the right type to us combine_eltypes is
# this function and it require Broadcasted42 to be inferrable
broadcast_eltype(bc::Broadcasted42) = Base.Broadcast.combine_eltypes(bc.f, bc.args)
broadcast_eltype(A) = Base.eltype(A)  # Tuple, Array, etc.

# To use combine_eltypes, we also need to redefine this function with our own Broadcasted object 
Base.Broadcast._broadcast_getindex_eltype(bc::Broadcasted42) = (
    Base.Broadcast.combine_eltypes(bc.f, bc.args)
)

# ═══════════════════════════════════ obtain broadcasted value ═══════════════════════════════════ #

# Because Broadcasted42 is a lazy types, the values are computed only when calling getindex

# ──── apply broadcasted function ──────────────────────────────────────────────────────────────── #

@inline function Base.getindex(bc::Broadcasted42{ScalarStyle})
    args = _eval_broadcast_args(bc.args, 1)
    return bc.f(args...)
end

@inline function Base.getindex(bc::Broadcasted42, I::BroadcastIndex)
    @boundscheck checkbounds(bc, I)
    args = _eval_broadcast_args(bc.args, I)
    Base.Broadcast
    return bc.f(args...)
end

# ──── evaluate args ───────────────────────────────────────────────────────────────────────────── #

@propagate_inbounds _eval_broadcast_args(A::Tuple, I) = (
    ntuple(j -> _eval_broadcast_args(A[j], I), length(A))
)

@propagate_inbounds _eval_broadcast_args(x::AbstractArray,   I) = x[I]
@propagate_inbounds _eval_broadcast_args(x::AbstractNDArray, I::CartesianIndex42) = x[I]
@propagate_inbounds _eval_broadcast_args(x::AbstractArray,   I::CartesianIndex42) = (
    x[convert(CartesianIndex, I)]
)

@propagate_inbounds _eval_broadcast_args(bc::Broadcasted42, I) = bc[I]

# Scalars just ignore I
@propagate_inbounds _eval_broadcast_args(x::ScalarArgs, I) = x[]

# ═════════════════════════════════════════ materialize ══════════════════════════════════════════ #

# Materialize is the function used to evaluate the lazy Broadcasted object


# ──── materialize functions ───────────────────────────────────────────────────────────────────── #
            
materialize42(bc::Broadcasted42{ScalarStyle}) = bc[]

# Create the ouput container, which will evaluate it's content in the copy! function
function materialize42(bc::Broadcasted42{Style, Shape}) where {Style, Shape}
    ElType = broadcast_eltype(bc)
    dest   = similar(bc, ElType, Shape)
    return copy!(dest, bc)
end

# ──── materialize! functions ──────────────────────────────────────────────────────────────────── #

# Those functions are used with in place operators such as +=, *=, ...

materialize42!(dest, bc::Broadcasted42) = copy!(dest, bc)

materialize42!(op::Op, dest, bc::Broadcasted42) where {Op <: InplaceOp} = copy!(op, dest, bc)

# ──── evaluate result ─────────────────────────────────────────────────────────────────────────── #

# The copy functions will evaluate the result when needed by calling getindex

function Base.copy!(dest,  bc::Broadcasted42{ScalarStyle})
    dest = bc[]
end

function Base.copy!(dest, bc::Broadcasted42)
    @boundscheck size(dest) == size(bc);
    for i in eachindex(bc)
        dest[i] = bc[i]
    end
    dest
end

function Base.copy!(op::Op, dest, bc::Broadcasted42) where {Op <: InplaceOp} 
    @boundscheck size(dest) == size(bc);
    for i in eachindex(bc)
        dest[i] = op(dest[i], bc[i])
    end
    dest
end
