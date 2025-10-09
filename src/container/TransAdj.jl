struct Transpose{T} <: AbstractMatrix{T}
    parent::NDArray{T, 2}

    Transpose(A::NDArray{T, 2}) where{T} = new{T}(A)
end

struct Adjoint{T} <: AbstractMatrix{T}
    parent::NDArray{T, 2}

    Adjoint(A::NDArray{T, 2}) where{T} = new{T}(A)
end


Adjoint(A::Matrix{T}) where{T} = Adjoint(A)


# The only difference between Transpose and Adjoint is seen if the type is complex, in this case the latter will also take the conjugate of the complex entries
const TransOrAdj = Union{Transpose, Adjoint}


# getters:

Base.getindex(a::TransOrAdj, indices::CopyIndices) = getindex(a.parent, indices)

Base.getindex(a::TransOrAdj, indices::Vararg{Int, 1}) = getindex(a.parent, indices...)

Base.getindex(a::TransOrAdj, indices::Vararg{T, 2}) where {T}  = getindex(a.parent, indices[2], indices[1])

# setters:

Base.setindex!(a::TransOrAdj, val, index::CartesianIndex{1}) = setindex!(a.parent, val, index)

Base.setindex!(a::TransOrAdj, val, index::CartesianIndex{2}) = setindex!(a.parent, val, CartesianIndex((index[2], index[1])))


# size:
Base.size(A::TransOrAdj, d::Integer) = size(A)[d]

Base.size(A::TransOrAdj) =  (size(A.parent, 2), size(A.parent, 1))

# similar:
Base.similar(A::TransOrAdj) = NDArray{eltype(A.parent)}(size(A))

# Adjoint behavior (in complex):

Base.getindex(a::Adjoint{Complex}, indices::CopyIndices) = conj( getindex(a.parent, indices) )

Base.getindex(a::Adjoint{Complex}, indices::Vararg{T, 2}) where {T}  = conj( getindex(a.parent, indices[2], indices[1]) )

# conjugate
conj(element::Number) = element

# conj ( a + ib ) = a - ib
conj(element::Complex) = Complex(element.real, -element.im)

function Base.copy!(dest::NDArray, A::TransOrAdj) 
   for (i, val) in enumerate(A)
      println("i: $i, val: $val")
      dest[i] = val
   end
end

# Iteration:

# First call
function Base.iterate(a::TransOrAdj)
    inds = CartesianIndices(size(a))   # all valid indices
    isempty(inds) && return nothing
    first_index = first(inds)
    return (a[first_index], (inds, 2))
end

# Subsequent calls
function Base.iterate(a::TransOrAdj, state)
    inds, i = state
    if i > length(inds)
        return nothing
    end
    idx = inds[i]
    return (a[idx], (inds, i + 1))
end