include("AbstractView.jl")

struct DroppedDimsView{T, N, ParentType <: AbstractNDArray{T}, IndicesType, Trivial} <: AbstractNDView{T, N, ParentType, IndicesType, Trivial}
    parent::ParentType
    indices::IndicesType
end



# Calculer la nouvelle forme (taille)
function Base.size(v::DroppedDimsView)
    # Filtrer les dimensions du parent qui sont de taille 1
    # On retourne un Tuple des dimensions qui NE SONT PAS de taille 1
    parent_dims = size(v.parent)
    new_dims = filter(d -> d != 1, parent_dims)
    return Tuple(new_dims)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     Simple functions:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

Base.parent(A::DroppedDimsView{T}) where T = parent(A.parent)

Base.pointer(A::DroppedDimsView{T}) where T = pointer(parent(A))

Base.IndexStyle(::Type{DroppedDimsView}) = IndexCartesian()




# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     dropdims function:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Cette méthode construit la DroppedDimsView
function Base.dropdims(c::AbstractNDArray; dims::Union{Nothing, Integer, Tuple}=nothing)
    
    # Si dims est non spécifié, on retire toutes les dimensions unitaires.
    if dims === nothing
        dims_to_drop = Tuple(i for (i, d) in enumerate(size(c)) if d == 1)
    
    # Si dims est un scalaire ou un tuple de dimensions fournies par l'utilisateur.
    else
        # S'assurer que les dimensions demandées sont bien de taille 1 dans le AbstractNDArray
        dims_tuple = isa(dims, Integer) ? (dims,) : dims
        for d in dims_tuple
            if size(c, d) != 1
                throw(DimensionMismatch("Dimension $d demandée pour dropdims mais sa taille est $(size(c, d)), pas 1."))
            end
        end
        dims_to_drop = Tuple(dims_tuple)
    end
    
    # Si rien n'est à retirer, retourner la vue originale
    if isempty(dims_to_drop)
        return c
    end

    # Créer et retourner la nouvelle vue paresseuse
    N_in = ndims(c)
    N_out = N_in - length(dims_to_drop)
    parent_dims = size(c)
    new_dims = filter(d -> d != 1, parent_dims)
    return DroppedDimsView{eltype(c), N_in, N_out, typeof(c)}(c, new_dims, dims_to_drop)
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Get and set element:                                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Scalar
@propagate_inbounds _getElement(a::ScalarNDView{T}) where {T} = (
    content = parent(a); 
    content[_getIndex(a, ())]
)

@propagate_inbounds _setElement!(a::ScalarNDView{T}, val::T) where {T} = (
    content = parent(a);
    content[_getIndex(a)] = val
)

# Cartesian index
@propagate_inbounds _getElement(a::AbstractNDView{T, N}, I::CartesianIndex{N}) where {T, N} = (
    content = parent(a);
    content[_getIndex(a, I)]
)

@propagate_inbounds _setElement!(a::AbstractNDView{T, N}, val::T, I::CartesianIndex{N}) where {T, N} = (
    content = parent(a);
    content[_getIndex(a, I)...] = val
)

# 'Normal' index
@propagate_inbounds _getElement(a::AbstractNDView{T, N}, indices::NTuple{N, Any}) where {T, N} = ( 
    content = parent(a); 
    content[_getIndex(a, indices)...] 
)

@propagate_inbounds _setElement!(a::AbstractNDView{T, N}, val::T, indices::NTuple{N, Any}) where {T, N} = (
    content = parent(a);
    content[_getIndex(a, indices)...] = val
)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       _getIndex:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


@propagate_inbounds _getIndex(::DroppedDimsView{DType,  N_in, N_out, P}) where  {DType, N_in, N_out, P} = CartesianIndex(ntuple(_ -> 1, N_in))

@propagate_inbounds _getIndex(a::DroppedDimsView{DType, N_in, N_out}, I::CartesianIndex{N_out}) where {DType, N_in, N_out} = _getIndex(a, Tuple(I))

@propagate_inbounds function _getIndex(a::DroppedDimsView{DType, N_in, N_out, P}, indices::NTuple{N_out, Any}) where  {DType, N_in, N_out, P}
    parent_indices = []
    # Indexation actuelle dans la vue résultante (indices...)
    output_pos = 1 
    
    # Parcourir toutes les dimensions du parent original
    for d in 1:N_in
        if d in a.dropped_dims
            # Si la dimension était retirée (taille 1), son index dans le parent est toujours 1.
            push!(parent_indices, 1)
        else
            # Sinon, son index est fourni par l'indexation de la vue (indices...)
            push!(parent_indices, indices[output_pos])
            output_pos += 1
        end
    end
  #  println("indices: ", parent_indices)
    return parent_indices
end