# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                           PaddedShape:                                           #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# This object is used as a view on tuple to prepend them by one without modifying them

struct PaddedShape{OriginalDimension, PaddedDimension}
    originalShape::NTuple{OriginalDimension,Int}
end

# ──── constructors ────────────────────────────────────────────────────────────────────────────── #

function PaddedShape(originalShape::NTuple{N,Int}, ::Val{maxDim}) where {N, maxDim}
    PaddedShape{N, maxDim}(originalShape)
end

function PaddedShape(originalArray::AbstractArray{T,N}, ::Val{maxDim}) where {T,N, maxDim}
    PaddedShape{N, maxDim}(size(originalArray))
end

function padded_shapes(arrays::Tuple)
    max_dim_val = maximum(map(ndims, arrays))            # Runtime integer value
    return _padded_shapes_impl(arrays, Val(max_dim_val)) # Types stable
end

function _padded_shapes_impl(arrays::Tuple, ::Val{MaxDim}) where {MaxDim}
    return map(arr -> PaddedShape(size(arr), Val(MaxDim)), arrays)
end

# ──── getindex ────────────────────────────────────────────────────────────────────────────────── #

function Base.getindex(prep_shape::PaddedShape{OriginalDimension, expectedDimension}, 
                       i::Index) where {OriginalDimension, expectedDimension}
    if (expectedDimension == OriginalDimension)
        return prep_shape.originalShape[i]
    else
        offset = expectedDimension - OriginalDimension
        return (i <= offset) ? 1 : prep_shape.originalShape[i-offset]
    end
end

# ──── core properties ─────────────────────────────────────────────────────────────────────────── #

Base.length(padded_shape::PaddedShape{OriginalDimension, expectedDimension}) where {
    OriginalDimension, expectedDimension} = expectedDimension

Base.axes(padded_shape::PaddedShape{OriginalDimension, PaddedDimension}) where {
    OriginalDimension, PaddedDimension} = map(Base.OneTo, padded_shape)

Base.size(padded_shape::PaddedShape{OriginalDimension}) where {
    OriginalDimension} = map(length, axes(padded_shape))

original_dim(padded_shape::PaddedShape{OriginalDimension}) where {
    OriginalDimension} = OriginalDimension


# ──── print function ──────────────────────────────────────────────────────────────────────────── #

function Base.show(io::IO, p::PaddedShape)
    N = length(p)
    if N == 0
        print(io, "PaddedShape(())")
    else
        result = "(" * string(p[1])

        for i in 2:length(p)
            result *= ", $(string(p[i]))"
        end

        result *= " )"
        print(io, result)
    end
end

Base.show(io::IO, ::MIME"text/plain", p::PaddedShape) = show(io, p)