# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       Constant:                                               #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# ======== Indices Types ====================================================================== #

# Return a single elements of the array ( if enough are provided )
const Index = Int                          

# Return a view if:
# - a colon or a range is present
const ViewIndices = Union{AbstractRange, Int, Colon, Base.Slice} 

# Fancy indices are arrays of integer or boolean
const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# Return a copy of the array, if at least one fancy index is present
const CopyIndices = Union{AbstractRange, Int, Colon, Base.Slice, AbstractArray{Int}, AbstractArray{Bool}}

# ======== Other aliases ====================================================================== #

const AllCartesianIndex{N} = Base.AbstractCartesianIndex{N}

# Range with only int for first and last
const OrdinalRangeInt = OrdinalRange{Int,Int}

# A tuple of ranges
const IndicesRange{N} = NTuple{N,OrdinalRangeInt}

# Values to differentiate col major from row major
const ColOrder = 1
const RowOrder = 2

# ======== Broadcasting ====================================================================== #

struct BroadcastedFunction42{F <: Function} <: Function
    f::F
end 

@inline (op::BroadcastedFunction42)(x...; kwargs...) = op.f.(x...; kwargs...)

const Operator = Union{typeof(+), typeof(-), typeof(*), typeof(/), typeof(<), typeof(>), BroadcastedFunction42}

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
