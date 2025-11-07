include("macro.jl")
include("Errors.jl")

# ──── index types ─────────────────────────────────────────────────────────────────────────────── #

# linear
const Index = Int

# Cartesian
const IndicesInt{N}        = NTuple{N, Index}
const AllCartesianIndex{N} = Base.AbstractCartesianIndex{N}
const Indices{N}           = Union{NTuple{N, Index}, AllCartesianIndex{N}}

# Return a view:
const ViewIndices = Union{AbstractRange, Int, Colon, Base.Slice} 

# Fancy indices are arrays of integer or boolean
const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# Return a copy:
const CopyIndices = Union{
    AbstractRange, 
    Int, 
    Colon, 
    Base.Slice,
    AbstractArray{Int}, 
    AbstractArray{Bool}
}

# ──── other aliases ───────────────────────────────────────────────────────────────────────────── #

const ZeroDimIndex = Union{Tuple{}, Nothing}

const OrdinalRangeInt = OrdinalRange{Int,Int}

const IndicesRange{N} = NTuple{N,OrdinalRangeInt}

const ColOrder = 1
const RowOrder = 2

const NestedArray{T} = AbstractArray{T, 1}

const MultiDimArray{T, N} = AbstractArray{T, N}


# ═════════════════════════════════════════ broadcasting ═════════════════════════════════════════ #
        
# ──── BroadcastedFunction ─────────────────────────────────────────────────────────────────────── #

struct BroadcastedFunction42{F <: Function} <: Function
    f::F
end 

@inline (op::BroadcastedFunction42)(x...; kwargs...) = op.f.(x...; kwargs...)

# ──── other ───────────────────────────────────────────────────────────────────────────────────── #

const Operator = Union{
    typeof(+), 
    typeof(-), 
    typeof(*), 
    typeof(/), 
    typeof(<), 
    typeof(>),
    BroadcastedFunction42
}

const BroadcastIndex = Union{ Integer, AllCartesianIndex }

const ScalarArgs = Union{AbstractArray{<:Any,0}, Number}

const InplaceOp = Union{typeof(+) , typeof(-), typeof(*), typeof(/)}

const op_map = Dict(
    :.> => >,
    :.< => <,
    :.+= => +,
    :.-= => -,
    :.*= => *,
    :./= => /,
    :.+ => +,
    :.- => -,
    :.* => *,
    :./ => /,
    :.= => nothing
)
