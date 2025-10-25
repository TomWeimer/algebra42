
# Padded Shape:
# ------------------

# Structure used when prepending shapes with ones
struct PaddedShape{OriginalDimension}
    originalShape::Ref{NTuple{OriginalDimension,Int}}
    expectedDimension::Ref{Int}
end

function PaddedShape(originalShape::NTuple{N,Int}, maxDim::Ref{Int}) where {N}
    # If the max dimension is smaller than the dimension of original shape then update it
    if (maxDim[] < N)
        maxDim[] = N
    end
    # Return a "view" on the original shape
    PaddedShape{N}(Ref(originalShape), maxDim)
end

function PaddedShape(originalArray::AbstractArray{T,N}, maxDim::Ref{Int}) where {T,N}
    # If the max dimension is smaller than the dimension of the original array then update it
    if (maxDim[] < N)
        maxDim[] = N
    end
    # Return a "view" on the shape of the array
    PaddedShape{N}(Ref(size(originalArray)), maxDim)
end

# Using this function and an offset we mimic a shape prepended with 1 without allocating it
function Base.getindex(prep_shape::PaddedShape{OriginalDimension}, i::Int) where {OriginalDimension}
    if (prep_shape.expectedDimension[] == OriginalDimension)
        return prep_shape.originalShape[][i]
    else
        offset = prep_shape.expectedDimension[] - OriginalDimension
        return (i <= offset) ? 1 : prep_shape.originalShape[][i-offset]
    end
end

# Return the original dimension of the shape
original_dim(padded_shape::PaddedShape{OriginalDimension}) where {OriginalDimension} = OriginalDimension

Base.length(padded_shape::PaddedShape{OriginalDimension}) where {OriginalDimension} = padded_shape.expectedDimension[]

Base.axes(padded_shape::PaddedShape{OriginalDimension})where {OriginalDimension} = map(Base.OneTo, padded_shape)

Base.size(padded_shape::PaddedShape{OriginalDimension}) where {OriginalDimension} = map(length, axes(padded_shape))


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