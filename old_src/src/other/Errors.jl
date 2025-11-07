module Errors
export ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY, ERR_IDX_TYPE_VECTOR, ERR_CARTESIAN_SETINDEX, ERR_PERM_IPERM_TYPE, ERR_PERM_IPERM_INV, ERR_PERM_INVALID, ERR_PERM_DIMS,  ERR_SCALAR_DIAG, ERR_NOT_STORED_VALUE, ERR_IS_NOT_SQUARE,
       ERR_DIM_HEIGHT, ERR_DIM_WIDTH

const ERR_INDEX_INT_ON_SCALAR_ARRAY = "Cannot index a 0-dimensional view with a single integer. Use `view[()]` or `view.item()`."

const ERR_EMPTY_TUPLE_ON_ND_ARRAY = "Empty tuple indexing is only valid for 0-dimensional arrays."

const ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY = "Cannot use `:` or ranges to index a 0-dimensional view. Use `view[()]` or `view.item()`."

const ERR_IDX_TYPE_VECTOR = "Error invalid index types for 1 dimensional array"

const ERR_CARTESIAN_SETINDEX = "Cannot assign to a cartesian indices — values are generated, not stored."

const ERR_PERM_IPERM_TYPE = "both perm and iperm must both be NTuple{N, Int}"

const ERR_PERM_IPERM_INV = "iperm and perm must be inverses"

const ERR_PERM_INVALID = "the permutation is not valid"

const ERR_PERM_DIMS = "destination AbstractNDArray of incorrect size"

const ERR_SCALAR_DIAG = "It doesn't make sense to consider a scalar as a DiagonalArray"

const ERR_NOT_STORED_VALUE = "Values are computed not stored in memory so it doesn't make sense to set them"

const ERR_IS_NOT_SQUARE = "The dimensions of the array are not all equals"

const ERR_DIAGONAL_DIM = "The size of the diagonal given do not match the size of the array"

const ERR_DIM_HEIGHT = "The arrays have not the same height"

const ERR_DIM_WIDTH = "The arrays have not the same width"

const ERR_VIEW_INDICES_PARENT_DIM = "the number of indices used when creating a view must match the parent dimensionality"
end

