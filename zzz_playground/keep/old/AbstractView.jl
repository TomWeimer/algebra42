include("AbstractNDArray.jl")

# AbstractNDView is a type that regroups all contingous view of the array they are:
# NDArrayView, ReshapedNDArray, Transpose

abstract type AbstractNDView{T, N, ParentType <: AbstractArray{T}, IndicesType, LinearIndex} <: AbstractNDArray{T, N} end

const AbstractScalarView{T, P, I, L} = AbstractNDView{T, 0, P, I, L}
const AbstractScalarViewFast{T, N, P, I} = AbstractNDView{T, 0, P, I, true}
const AbstractScalarViewSlow{T, N, P, I} = AbstractNDView{T, 0, P, I, false}


const NDFastView{T, N, P, I} = AbstractNDView{T, N, P, I, true} 
const NDSlowView{T, N, P, I} = AbstractNDView{T, N, P, I, false} 


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Get and set element:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
@propagate_inbounds _getElement(a::AbstractScalarView{T}) where {T} = (
    parent = Base.parent(a); 
    parent[_getIndex(IndexStyle(a), a)]
)

@propagate_inbounds _setElement!(a::AbstractScalarView{T}, val::T) where {T} = (
    parent = Base.parent(a);
    parent[_getIndex(IndexStyle(a), a)] = val
)

# Linear index
@propagate_inbounds _getElement(a::AbstractNDView{T, N}, i::Int) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, i)]
)

@propagate_inbounds _setElement!(a::AbstractNDView{T, N}, val::T, i::Int) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, i)] = val
)


# Cartesian index
@propagate_inbounds _getElement(a::AbstractNDView{T, N}, I::CartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, I)]
)

@propagate_inbounds _setElement!(a::AbstractNDView{T, N}, val::T, I::CartesianIndex{N}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, I)] = val
)

# 'Normal' index
@propagate_inbounds _getElement(a::AbstractNDView{T, N}, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a); 
    parent[_getIndex(a, indices)] 
)

@propagate_inbounds _setElement!(a::AbstractNDView{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
    parent = Base.parent(a);
    parent[_getIndex(a, indices)] = val
)



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Linear Index:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
# TODO: a changer
@propagate_inbounds _getIndex(a::AbstractScalarViewFast) = 1

# Linear index
@propagate_inbounds _getIndex(::IndexLinear, a::AbstractNDView{T, N}, i::Int) where {T, N} = i

# Cartesian index
@propagate_inbounds _getIndex(a::NDFastView{T, N}, I::CartesianIndex{N}) where {T, N} = offset(a, I)

# 'Normal' index
@propagate_inbounds _getIndex(a::NDFastView{T, N}, indices::NTuple{N, Any}) where {T, N} = offset(a, indices)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Obtain Cartesian Index:                                 #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

@propagate_inbounds _getIndex(a::AbstractScalarViewSlow) = ntuple(_ -> 1, Base.ndims(parent(a)))

# Linear index
@propagate_inbounds _getIndex(a::NDSlowView{T, N}, i::Int) where {T, N} = fromLinearToCartesian(i, a)

# Cartesian index
@propagate_inbounds _getIndex(a::NDSlowView{T, N}, I::CartesianIndex{N}) where {T, N} = _toParentCartesian(a, I, Base.ndims(parent(a)))

# 'Normal' index
@propagate_inbounds _getIndex(a::NDSlowView{T, N}, indices::NTuple{N, Int}) where {T, N} = _toParentCartesian(a, CartesianIndex(indices), Base.ndims(parent(a)))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                from fromLinearToCartesian:                                    #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

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
    
    return CartesianIndex(_toParentIndices(indices, a.indices, Base.ndims(parent(a)))...)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                from Cartestian to parent Cartesian:                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# works only for view
function _toParentIndices(cinds, sliced_indices::AbstractArray{T, N}, N_parent::Int) where{T, N}
    finalIndices = []

    cpos = 1
    for i in 1:N_parent
        idx = sliced_indices[i]
        
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
function _toParentCartesian(a::AbstractNDArray{T, N}, I::CartesianIndex{N}, N_parent::Int) where{T, N}
    N_parent == N && return I

    finalIndices = []
    
    cpos = 1

    for i in 1:N_parent
        idx = a.indices[i]
        
        if idx isa Integer
            push!(finalIndices, idx)
        else
            val_idx = I[cpos]
            push!(finalIndices, idx[val_idx])
            cpos += 1
        end
    end
    return CartesianIndex(finalIndices...)
end