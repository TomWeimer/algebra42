include("AbstractNDArray.jl")

# AbstractNDView is a type that regroups all contingous view of the array they are:
# NDArrayView, ReshapedNDArray, Transpose

abstract type ContingousNDArray{T, N} <: AbstractNDArray{T, N} end

const ContingousScalar{T} = ContingousNDArray{T, 0}

# Return the strides of thos view
Base.strides(a::ContingousNDArray) = a.strides
Base.stride(a::ContingousNDArray, k::Integer) = ( @boundscheck checkindex(Bool, a, k); @inbounds a.strides[k])

# modify and access elements:

# Scalar
@propagate_inbounds _getElement(a::ContingousScalar{T}) where {T} = (
    parent = parent(a); 
    parent[_getIndex(IndexStyle(a), a)]
)

@propagate_inbounds _setElement!(a::ContingousScalar{T}, val::T) where {T} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a)] = val
)

# Linear index
@propagate_inbounds _getElement(a::ContingousNDArray{T, N}, i::Int) where {T, N} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a, i)]
)

@propagate_inbounds _setElement!(a::ContingousNDArray{T, N}, val::T, i::Int) where {T, N} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a, i)] = val
)


# Cartesian index
@propagate_inbounds _getElement(a::ContingousNDArray{T, N}, I::CartesianIndex{N}) where {T, N} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a, I)]
)

@propagate_inbounds _setElement!(a::ContingousNDArray{T, N}, val::T, I::CartesianIndex{N}) where {T, N} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a, I)] = val
)

# 'Normal' index
@propagate_inbounds _getElement(a::ContingousNDArray{T, N}, indices::NTuple{N, Any}) where {T, N} = ( 
    parent = parent(a); 
    parent[_getIndex(IndexStyle(a), a, indices)] 
)

@propagate_inbounds _setElement!(a::ContingousNDArray{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
    parent = parent(a);
    parent[_getIndex(IndexStyle(a), a, indices)] = val
)



# Obtain linear index:
# --------------------

# Scalar
# TODO: a changer
@propagate_inbounds _getIndex(::IndexLinear, a::ContingousScalar{T}) where {T} = 1

# Linear index
@propagate_inbounds _getIndex(::IndexLinear, a::ContingousNDArray{T, N}, i::Int) where {T, N} = i

# Cartesian index
@propagate_inbounds _getIndex(a::ContingousNDArray{T, N}, I::CartesianIndex{N}) where {T, N} = _offset(a, I)

# 'Normal' index
@propagate_inbounds _getIndex(a::ContingousNDArray{T, N}, indices::NTuple{N, Any}) where {T, N} = _offset(a, indices)

# Obtain cartesian index:
# ----------------------

@propagate_inbounds _getIndex(::IndexCartesian, a::ContingousScalar{T}) where {T} = ntuple(_ -> 1, ndims(a.parent))

# Linear index
@propagate_inbounds _getIndex(::IndexLinear, a::ContingousNDArray{T, N}, i::Int) where {T, N} = fromLinearToCartesian(i, a)

# Cartesian index
@propagate_inbounds _getIndex(a::ContingousNDArray{T, N}, I::CartesianIndex{N}) where {T, N} = _toParentCartesian(a, I)

# 'Normal' index
@propagate_inbounds _getIndex(a::ContingousNDArray{T, N}, indices::NTuple{N, Int}) where {T, N} = _toParentCartesian(a, CartesianIndex(indices))


# Scalar
_offset(a::ContingousNDArray{T, 0}) where T = 1

# CartesianIndex
_offset(a::ContingousNDArray{T, N}, I::CartesianIndex{N}) where {T, N} = _offset(a, Tuple(I))

# 'Normal' indices 
function _offset(a::ContingousNDArray{T, N}, indices::NTuple{N, Int}) where {T, N}
    offset = 0
    strides = strides(a)
    for i in 1:N
         offset += (indices[i] - 1) * strides[i]
    end
    return offset + 1
end

# Index de base : On commence par $L' = L - 1$ (pour travailler en 0-basé).
# Itération : Pour chaque dimension $d$ de 1 à $N_V$ 
# - Taille : Récupérer la taille $s_d$ de la dimension $d$ (c'est-à-dire $size(V, d)$).
# - Index id​ :$$i_d = 1 + L' \pmod{s_d}$$
# - Mise à jour :$$L' = \lfloor L' / s_d \rfloor$$
# Résultat : Les indices $(i_1, i_2, \dots, i_{N_V})$ sont les indices cartésiens dans la vue $V$.


# works only for view
function fromLinearToCartesian(L::Int, a::AbstractNDArray{T, N}) where {T, N}
    L_0 = L - 1

    indices = []

    for d in 1:N
        s_d = size(a, d)
        i_d = Base.mod(L_0, s_d) 
        push!(indices, i_d + 1)
        L_0 = div(L_0, s_d) # division entiere
    end

    
    return CartesianIndex(_toParentIndices(indices, a)...)
end

# works only for view
function _toParentIndices(cinds, a::AbstractNDArray{T, N}) where{T, N}
    finalIndices = []

    N_parent = ndims(parent(a))
    cpos = 1
    for i in 1:N_parent
        idx = a.indices[i]
        
        if idx isa Integer
            push!(finalIndices, idx)
        else
            cindex = cinds[cpos]
            push!(finalIndices, idx[cindex])
            cpos += 1
        end
    end
    return finalIndices
end



# works only for view
function _toParentCartesian(a::AbstractNDArray{T, N}, I::CartesianIndex{N}) where{T, N}
    length(I) <= N || throw(ArgumentError("Cartesian index given is to big"))
   
    N_parent = ndims(parent(a))

    N_parent == N && return I

    finalIndices = []

    cpos = 1
    for i in 1:N_parent
        idx = a.indices[i]
        
        if idx isa Integer
            push!(finalIndices, idx)
        else
            push!(finalIndices, I[cpos])
            cpos += 1
        end
    end
    return CartesianIndex(finalIndices...)
end


