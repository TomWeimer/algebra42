const AbstractUnitRangeInt = AbstractUnitRange{<:Integer}


struct LinearIndices{N,R<:NTuple{N, AbstractUnitRange{Int} }} <: AbstractArray{Int,N}
    indices::R
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Constructors:                                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Empty LinearIndices
LinearIndices(::Tuple{}) = LinearIndices{0,typeof(())}(())

# Convert to AbstractUnitRange{Int}
LinearIndices(inds::NTuple{N, AbstractUnitRange}) where {N} = LinearIndices( map(r -> convert(AbstractUnitRange{Int}, r), inds) )

# Convert to AbstractUnitRange{Int}
LinearIndices(inds::NTuple{N,Union{<:Integer, AbstractUnitRangeInt}}) where {N} = LinearIndices(map(_convert2ind, inds))


_convert2ind(i::Integer) = Base.oneto(i)

_convert2ind(ind::AbstractUnitRange) = first(ind):last(ind)

convert(::Type{LinearIndices{N,R}}, inds::LinearIndices{N}) where {N,R<:NTuple{N,AbstractUnitRange{Int}}} = LinearIndices{N,R}(convert(R, inds.indices))::LinearIndices{N,R}

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Promotion:                                          #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# function indices_promote_type(::Type{Tuple{R1,Vararg{R1,N}}}, ::Type{Tuple{R2,Vararg{R2,N}}}) where {R1,R2,N}
#     R = promote_type(R1, R2)
#     return Tuple{R,Vararg{R,N}}
# end

# promote_rule(::Type{LinearIndices{N,R1}}, ::Type{LinearIndices{N,R2}}) where {N,R1,R2} =
#     LinearIndices{N,indices_promote_type(R1, R2)}

# promote_rule(a::Type{Base.Slice{T1}}, b::Type{Base.Slice{T2}}) where {T1,T2} =
#     Base.el_same(promote_type(T1, T2), a, b)

# promote_rule(a::Type{Base.IdentityUnitRange{T1}}, b::Type{Base.IdentityUnitRange{T2}}) where {T1,T2} =
#     Base.el_same(promote_type(T1, T2), a, b)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                          Abstract Array Implementation:                                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Index Style:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# AbstractArray implementation
IndexStyle(::Type{<:LinearIndices}) = IndexLinear()


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Axes:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

axes(iter::LinearIndices) = map(Base.axes1, iter.indices)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Size:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

size(iter::LinearIndices) = map(length, iter.indices)

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           isassigned:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

isassigned(iter::LinearIndices, i::Int) = checkbounds(Bool, iter, i)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           getindex:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #
function getindex(iter::LinearIndices, i::Int)
    @inline
    @boundscheck checkbounds(iter, i)
    i
end

function getindex(iter::LinearIndices, i::AbstractRange{<:Integer})
    @inline
    @boundscheck checkbounds(iter, i)
    @inbounds isa(iter, LinearIndices{1}) ? iter.indices[1][i] : (first(iter):last(iter))[i]
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           copy:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

copy(iter::LinearIndices) = iter

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Iteration:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# More efficient iteration — predominantly for non-vector LinearIndices
# but one-dimensional LinearIndices must be special-cased to support OffsetArrays
iterate(iter::LinearIndices{1}, s...) = iterate(Base.axes1(iter.indices[1]), s...)
iterate(iter::LinearIndices, i=1) = i > length(iter) ? nothing : (i, i + 1)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       first/last:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Needed since firstindex and lastindex are defined in terms of LinearIndices
first(iter::LinearIndices) = 1
first(iter::LinearIndices{1}) = (@inline; first(Base.axes1(iter.indices[1])))
last(iter::LinearIndices) = (@inline; length(iter))
last(iter::LinearIndices{1}) = (@inline; last(Base.axes1(iter.indices[1])))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           show:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function show(io::IO, iter::LinearIndices)
    print(io, "LinearIndices(", iter.indices, ")")
end