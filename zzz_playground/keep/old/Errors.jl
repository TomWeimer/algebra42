module Errors
export ERR_INDEX_INT_ON_SCALAR_ARRAY, ERR_EMPTY_TUPLE_ON_ND_ARRAY, ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY

const ERR_INDEX_INT_ON_SCALAR_ARRAY = 
        "Cannot index a 0-dimensional view with a single integer. Use `view[()]` or `view.item()`."

const ERR_EMPTY_TUPLE_ON_ND_ARRAY = 
        "Empty tuple indexing is only valid for 0-dimensional arrays."

const ERR_RANGE_OR_COLON_ON_SCALAR_ARRAY =
         "Cannot use `:` or ranges to index a 0-dimensional view. Use `view[()]` or `view.item()`."

 end