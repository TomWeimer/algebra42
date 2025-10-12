# This fil contains dispatcher, which are optimized functions for specific task

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Collect only fancy indices:                                     #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Case 2: The first index is a regular index (Colon, Int, Range) -> ignore the first index and continue to filter the rest
_collect_fancy(::AbstractUnitRange, rest...) = _collect_fancy(rest...)

# Case 1: The first index is a fancy index (AbstractArray) -> keep the first index and continue to filter the rest
_collect_fancy(idx::AbstractArray, rest...) = (idx, _collect_fancy(rest...)...)

# Case 2: The first index is a regular index (Colon, Int, Range) -> ignore the first index and continue to filter the rest
_collect_fancy(idx::Union{Int, Colon, AbstractUnitRange}, rest...) = _collect_fancy(rest...)

# Final Case: There is no remaining indices, return an empty tuple
_collect_fancy() = ()


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Obtain dimensions from regular indices:                          #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# This colector create the regular shape


# Case 1: Scalar Index (Int) or Fancy Index (AbstractArray) (Do not particpate to the regular shape)
__collect_basic_dims(A_dims::Tuple, idx::Union{Int, AbstractArray}, rest_indices::Tuple, i::Int) = 
    __collect_basic_dims(A_dims, rest_indices..., i + 1)

# Case 2: Range participate
__collect_basic_dims(A_dims::Tuple, idx::AbstractUnitRange, rest_indices::Tuple, i::Int) = 
    (length(idx), __collect_basic_dims(A_dims, rest_indices..., i + 1)...)

# Case 3: Colon too
__collect_basic_dims(A_dims::Tuple, idx::Colon, rest_indices::Tuple, i::Int) = 
    (A_dims[i], __collect_basic_dims(A_dims, rest_indices..., i + 1)...)

# Final Case: There is no remaining indices, return an empty tuple
__collect_basic_dims(A_dims::Tuple, indices::Tuple, i::Int) = ()


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                               Obtain dimension from regular indices:                          #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

# Determines fancy indices:

# A fancy index is any AbstractArray (Vector, BitArray, etc.)
is_fancy(::AbstractRange) = false

is_fancy(::AbstractArray) = true

is_fancy(::Any) = false # Colon, Int, UnitRange, etc., are considered basic

is_fixed(::Int) = true # only single integer drop dimension
is_fixed(::Any) = false # Colon, AbstractArray, UnitRange, etc., are considered not fixed

# A basic index is NOT a fancy index
is_basic(idx) = !is_fancy(idx)

# Dispatcher:

# This dispatcher returns two states in contrast to the previous ones
# State 1: has_seen_fancy - Has an advanced index been encountered yet?
# State 2: has_seen_basic_after_fancy - Has a basic index been encountered after the first fancy index?


# Case A: Found a Fancy Index (First one, or consecutive) but has not seen yet a regular index
# Set has_seen_fancy=true and continue checking the remaining indices.
_check_for_separation(::Val{false}, ::Val{false}, idx::AbstractArray, rest...) = 
    _check_for_separation(Val{true}(), Val{false}(), rest...)

# Case A: Found a Fancy Index (First one, or consecutive) but has not seen yet a regular index
# Set has_seen_fancy=true and continue checking the remaining indices.
_check_for_separation(::Val{false}, ::Val{false}, idx::AbstractUnitRange, rest...) = 
    _check_for_separation(Val{true}(), Val{false}(), rest...)

# Case B: Found a regular Index but has not seen yet a fancy index
# State doesn't change yet, just recurse.
_check_for_separation(has_fancy, has_basic_after_fancy, idx::Union{Int, Colon, AbstractUnitRange}, rest...) =
    _check_for_separation(has_fancy, has_basic_after_fancy, rest...)

# Case E: 
_check_for_separation(::Val{true}, ::Val{false}, idx::AbstractArray, rest...) =
    _check_for_separation(Val{true}(), Val{false}(), rest...)

_check_for_separation(::Val{true}, ::Val{false}, idx::AbstractUnitRange, rest...) =
    _check_for_separation(Val{true}(), Val{true}(), rest...)

# Case C: Found a Basic Index after a fancy index
# Switch the has_seen_basic_after_fancy flag to true and recurse.
_check_for_separation(::Val{true}, ::Val{false}, idx::Union{Int, Colon, AbstractUnitRange}, rest...) =
    _check_for_separation(Val{true}(), Val{true}(), rest...)

# Case D: SEPARATION DETECTED!
# We've seen a fancy index AND we've seen a basic index after it. If the current index is FANCY, we have separation.
_check_for_separation(::Val{true}, ::Val{true}, idx::AbstractArray, rest...) = true

# Case G: SEPARATION DETECTED!
# We've seen an integer AND we've seen a basic index after it. If the current index is FANCY, we have separation.
_check_for_separation(::Val{true}, ::Val{true}, idx::Int, rest...) = true

# Final Case: End of recursion: No separation found.
_check_for_separation(::Val, ::Val) = false

function fancy_indices_are_together(indices::Tuple)
    # Start the recursive check
    return !_check_for_separation(Val{false}(), Val{false}(), indices...)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#               Obtain output dimensions for unseparrated fancy indices:                        #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


# Base case: no more indices left
_collect_out_dims(::Tuple, ::Tuple, i::Int, ::Val) = ()

# Case A: Colon (:) — before fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple, i::Int, ::Val{false}, idx::Colon, rest_indices...) =
    (A_dims[i], _collect_out_dims(S_fancy, A_dims, i + 1, Val(false), rest_indices...)...)

# Case B: Range — before fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple, i::Int, ::Val{false}, idx::AbstractUnitRange, rest_indices...) =
    (length(idx), _collect_out_dims(S_fancy, A_dims, i + 1, Val(false), rest_indices...)...)

# Case C: Fancy index (array) — first fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple,  i::Int, ::Val{false}, idx::AbstractArray, rest_indices...) =
    (S_fancy..., _collect_out_dims(S_fancy, A_dims, i + 1, Val(true), rest_indices...)...)

# Case D: Fancy index — after fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple,  i::Int, ::Val{true}, idx::AbstractArray, rest_indices...) =
    _collect_out_dims(S_fancy, A_dims,  i + 1, Val(true), rest_indices...)

# Case E: Colon (:) — after fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple, i::Int, ::Val{true},  idx::Colon, rest_indices...) =
    (A_dims[i], _collect_out_dims(S_fancy, A_dims, i + 1, Val(true), rest_indices...)...)

# Case F: Range — after fancy
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple, i::Int, ::Val{true}, idx::AbstractUnitRange, rest_indices...) =
    (length(idx), _collect_out_dims(S_fancy, A_dims,  i + 1, Val(true), rest_indices...)...)

# Optional: Case G — scalar integer index (removes one dimension)
_collect_out_dims(S_fancy::Tuple, A_dims::Tuple, i::Int, flag::Val, idx::Int, rest_indices...) =
    _collect_out_dims(S_fancy, A_dims,  i + 1, flag, rest_indices...)

# Entry point
function obtain_output_shape(S_fancy::Tuple, shape_A::Tuple, indices::Tuple)
    return _collect_out_dims(S_fancy, shape_A, 1, Val(false), indices...)
end



# This dispatcher is required because numpy handles differently the advanced indexing depending on two scenario:

# Scenario 1: All Advanced Indices are Together
#  If all fancy (advanced) indices are contiguous (no slices, colons, or basic integers separating them), 
#  broadcasted shape created from the fancy indices replace their place in the array of indices.
#   - example: A[:, Fancy1, Fancy2] or A[Fancy1, Fancy2, :]  or A[:, Fancy1, Fancy2, :]

# Scenario 2: Advanced Indices are separated
#  Otherwise if the fancy indices are separated, they are first broadcasted together this shaoe is S_fancy
#  then the shape of the other regular index create S_regular
#  In this second scenerio the output shape is then S_out = (S_fancy, S_regular)
#   - example: A[:, Fancy1, :, Fancy2] or A[Fancy1, Fancy2, :, 1]  or A[Fancy1, Fancy2, :, Fancy3]

# Note: Scalars do not particpate in the broadcasting that determines the of the fancy dimensions
#           - example: A with shape (6, 5, 4, 3, 2)  and accesed as A[:, :, 1, :, :] return a view with shape (6, 5, 3, 2 ) 
#      However they participate in the reduction of the dimension of the basic shape
#       - example: B with shape (6, 5, 4, 3, 2)  and accesed as B[[1; 3; 4], [1, 3], :, 1, :] give the following
#                  1. The fancy indices [1; 3; 4] with shape (3, 1)  and [1, 3] with shape (2,) are broadcasted thus their shapes becomes
#                     (3, 1)  and (1, 2) and are merged to S_fancy = (3, 2)
#                  2. The shape of the other indices (without the scalars) are the same as the original's so it is S_regular = (4, 2)
#                  3. Now the final shape becomes (S_fancy, S_regular) so it is (3, 2, 4, 2) notice that the scalar didn't participate
#                     to the shape because the dimension was simply discarded
