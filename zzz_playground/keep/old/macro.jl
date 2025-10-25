# Check that the dimension of the first element is the biggest 
# Macro pour vérifier que tous les tableaux suivants ont EXACTEMENT la même taille
# que le premier tableau (arrs[1]).
macro checksize(arrs...)
    if length(arrs) < 2
        return :(nothing)
    end

     # Escape variable names to preserve correct scope
    arrs = [esc(a) for a in arrs]

    # 1. Construction de l'expression de vérification (à exécuter au runtime)
    # Exemple : size(A) == size(B) && size(A) == size(C)
    checks = [:( size($(arrs[1])) == size($(arrs[i])) ) for i in 2:length(arrs)]
    combined_check = foldl((a,b)->:(a && $b), checks)

    # 2. Construction de l'expression du message d'erreur (à exécuter au runtime)
    
    # Fonction locale pour créer l'expression "varname = value"
    function size_info_expr(arr_sym)
        arr_name = string(arr_sym) 
        # L'expression résultante sera : "A = " * string(size(A))
        return :($arr_name * " = " * string(size($arr_sym))) 
    end

    # Créer un tableau d'expressions pour toutes les informations de taille
    size_info_expressions = [size_info_expr(arr) for arr in arrs]

    # Créer l'expression qui joint toutes les informations avec ", "
    join_call_expr = :(join([$(size_info_expressions...)], ", "))

    # Construction de l'expression finale du message : "Size mismatch: " + joined_info
    msg_expr = :( "Size mismatch: " * $join_call_expr )

    # 3. Retourner l'expression @assert finale.
    # On interpole l'expression de vérification ($combined_check) et l'expression du message ($msg_expr).
    return :( @assert $combined_check $msg_expr )
end

"""
    @mustoverload

Use on the RHS of a function definition to throw NotImplementedError.
"""
macro mustoverload()
    return :(throw(NotImplementedError("This function must be overloaded for concrete subtypes")))
end


macro checknindices(a, indices)
    quote
        length($(esc(indices))) <= ndims($(esc(a))) ||
            throw(ArgumentError("too many indices ($(length($(esc(indices))))) for $(ndims($(esc(a))))-dimensional array"))
    end
end

macro throw_index_error(indices, msg)
    return :(throw(DomainError($indices, $msg)))
end

function expand_indices(indices, N)
    ntuple(i -> i <= length(indices) ? indices[i] : Colon(), N)
end

# # to overload if needed
# @propagate_inbounds _getIndex(a::AbstractNDArray{DType, N}, indices::NTuple{N, Any}) where {DType, N} = indices

# # to overload if needed
# @propagate_inbounds _getIndex(a::AbstractNDArray{DType, N}, I::CartesianIndex{N}) where {DType, N} = I

# # to overload if needed
# @propagate_inbounds _getIndex(a::AbstractNDArray{DType, N}) where {DType, N} = ntuple(_ -> 1, N)

# function Base.copy(src_array::AbstractNDArray{DType, NDim}) where {DType, NDim}
#     # to change back
#     #  copy = (ndarray) ? NDArray{DType}(shape) : Array{DType, NDim}(undef, shape...)
#     dest_array = similar(src_array)
#     return copy!(dest_array, src_array)
# end

# function Base.copy!(dest_array::AbstractNDArray{DType, NDim}, src_array::AbstractNDArray{DType, NDim}) where {DType, NDim}
#     for (i, src_val) in enumerate(src_array)
#         dest_array[i] = src_val
#     end
#     return dest_array
# end

# function Base.copyto!(dest::AbstractArray, src::AbstractArray)
#     @checksize dest src
#     @inbounds for i in eachindex(dest, src)
#         dest[i] = src[i]
#     end
#     return dest
# end