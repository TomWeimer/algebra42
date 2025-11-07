module Errors
export ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, 
       ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY,ERR_IDX_TYPE_VECTOR, ERR_PERM_IPERM_TYPE, 
       ERR_PERM_IPERM_INV,ERR_PERM_INVALID, ERR_PERM_DIMS, ERR_SCALAR_DIAG, ERR_NOT_STORED_VALUE, 
       ERR_IS_NOT_SQUARE,ERR_DIAGONAL_DIM, ERR_DIM_HEIGHT, ERR_DIM_WIDTH, 
       ERR_VIEW_INDICES_PARENT_DIM

const ERR_INDEX_INT_ON_SCALAR_ARRAY =
    "Cannot index a 0-dimensional array with an integer except 1. Use `A[]` or `A[1]`."

const ERR_EMPTY_TUPLE_ON_ND_ARRAY =
    "Empty tuple indexing is only valid for 0-dimensional arrays."

const ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY =
    "Cannot use `:` or a range to index a 0-dimensional array. Use `view[]` or `getindex(view)`."

const ERR_IDX_TYPE_VECTOR =
    "Invalid index types for a 1-dimensional array."

const ERR_PERM_IPERM_TYPE =
    "`perm` and `iperm` must both be `NTuple{N, Int}`."

const ERR_PERM_IPERM_INV =
    "`iperm` must be the inverse of `perm`."

const ERR_PERM_INVALID =
    "Invalid permutation."

const ERR_PERM_DIMS =
    "Destination AbstractArray has incorrect size for the permutation."

const ERR_SCALAR_DIAG =
    "Cannot treat a scalar as a diagonal array."

const ERR_NOT_STORED_VALUE =
    "Values are computed, not stored, so assignment is not allowed."

const ERR_IS_NOT_SQUARE =
    "Array is not square — all dimensions must be equal."

const ERR_DIAGONAL_DIM =
    "Diagonal size does not match array size."

const ERR_DIM_HEIGHT =
    "Arrays do not have the same height."

const ERR_DIM_WIDTH =
    "Arrays do not have the same width."

const ERR_VIEW_INDICES_PARENT_DIM =
    "Number of indices used to create a view must match the dimensionality of the parent."
end