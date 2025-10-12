# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Symbol:                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

∑(arrayToSum) = sum(arrayToSum)


⋅(u::Vector{T}, v::Vector{T}) where {T} = dot(u, v)


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                           Addition:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Performs elementwise addition between two `NDArray` objects, following broadcasting rules.

# Arguments
- `ndarray1::NDArray`: The first array.
- `ndarray2::NDArray`: The second array.

# Returns
- `NDArray` containing elementwise sums.

# Complexity
- `O(N)` where `N` is the total number of elements in the broadcasted shape of the arrays.
"""
function add(ndarray1::NDArray, ndarray2::NDArray)
    return broadcast(add, ndarray1, ndarray2)
end

"""
Adds a scalar value to each element of the `NDArray`.

# Arguments
- `ndarray1::NDArray`: The array to which the scalar is added.
- `scalar::Number`: The scalar value.

# Returns
- `NDArray` with each element incremented by `scalar`.

# Complexity
- `O(N)` where `N` is the number of elements in `ndarray1`.
"""
function add(ndarray1::NDArray, scalar::Number)
    return broadcast(add, ndarray1, [scalar])
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                        Subtraction:                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Performs elementwise subtraction between two `NDArray` objects, following broadcasting rules.

# Arguments
- `ndarray1::NDArray`: The minuend array.
- `ndarray2::NDArray`: The subtrahend array.

# Returns
- `NDArray` containing elementwise differences.

# Complexity
- `O(N)` where `N` is the total number of elements in the broadcasted shape.
"""
function sub(ndarray1::NDArray, ndarray2::NDArray)
    return broadcast(sub, ndarray1, ndarray2)
end

"""
Subtracts a scalar from each element of the `NDArray`.

# Arguments
- `ndarray1::NDArray`: The array from which the scalar is subtracted.
- `scalar::Number`: The scalar value.

# Returns
- `NDArray` with each element decremented by `scalar`.

# Complexity
- `O(N)` where `N` is the number of elements in `ndarray1`.
"""
function sub(ndarray1::NDArray, scalar::Number)
    return broadcast(sub, ndarray1, [scalar])
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Scalar Product:                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Performs elementwise multiplication of an `NDArray` by a scalar.

# Arguments
- `ndarray1::NDArray`: The array to scale.
- `scalar::Number`: The scalar multiplier.

# Returns
- `NDArray` where each element is multiplied by `scalar`.

# Complexity
- `O(N)` where `N` is the number of elements in `ndarray1`.
"""
function prod(ndarray1::NDArray, scalar::Number)
    return broadcast(prod, ndarray1, [scalar])
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Linear Combination:                                     #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


"""
 Computes the **linear combination** of a set of vectors with given coefficients.

# Arguments
- `vectors::AbstractVector{Vector{T}}`: A vector of vectors `[v1, v2, ..., vk]` of the same length.
- `coefs::AbstractVector{T}`: A vector of coefficients `[λ1, λ2, ..., λk]`.

# Returns
- `Vector{T}` representing the linear combination:

result = λ_1 v_1 + λ_2 v_2 + ... + λ_k v_k

# Requirements
- All vectors in `vectors` must have the same length.
- `coefs` must have the same number of elements as `vectors`.

# Complexity
- Let `k` = number of vectors (`length(vectors)`)  
- Let `n` = dimension of each vector (`length(vectors[1])`)  

The function iterates over all vectors and multiplies each by its coefficient elementwise.  
- **Time Complexity:** `O(n * k)`  
- **Space Complexity:** `O(n)` for the result vector.
"""
function linear_combination(vectors::AbstractVector{Vector{T}}, coefs::AbstractVector{T}) where {T}
    n = length(vectors[1])

    check_linear_combination(vectors, coefs, n)

    result = zeroVector((n,), T)

    for (v, λ) in zip(vectors, coefs)
        result .+= λ .* v
    end

    return result
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     Linear Interpolation:                                     #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


"""
Computes the **linear interpolation** (lerp) between two arrays element-wise.

# Arguments
- `u::AbstractArray`: First array.
- `v::AbstractArray`: Second array (must have the same shape as `u`).
- `t::Number`: Interpolation factor in `[0, 1]`.  

# Returns
- An array of the same shape as `u` and `v`:
result[i] = (1 - t) * u[i] + t * v[i]

# Notes
- `t = 0` returns `u`.
- `t = 1` returns `v`.
- Works for any numeric array type, including `Float64`, `Int`, etc.

# Complexity
- Let `n` = number of elements in `u` (or `v`).  
- **Time Complexity:** `O(n)`  
- **Space Complexity:** `O(n)` for the result array.
"""
function lerp(u::V, v::V, t::Number) where {V<:AbstractArray}
    return (1 - t) .* u .+ t .* v
end

"""
Computes the **linear interpolation** between two scalar numbers.

# Arguments
- `u::Number`: First scalar.
- `v::Number`: Second scalar.
- `t::Number`: Interpolation factor in `[0, 1]`.

# Returns
- A single number:
text{result} = (1 - t) * u + t * v

# Complexity
- **Time Complexity:** O(1)  
- **Space Complexity:** O(1)
"""
function lerp(u::V, v::V, t::Number) where {V<:Number}
    return ((1 - t) * u) + (t * v)
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                     Dot Product:                                              #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the dot product of two vectors `u` and `v`.

# Arguments
- `u::Vector{T}`: First vector.
- `v::Vector{T}`: Second vector (must have same length as `u`).

# Returns
- Scalar of type `T` representing the dot product:  
  dot(u, v) = sum_i u[i] * v[i]

# Complexity
- Let `n = length(u)`  
- **Time Complexity:** O(n)  
- **Space Complexity:** O(1)
"""
function dot(u::Vector{T}, v::Vector{T}) where {T}
    length(u) == length(v) || throw(DimensionMismatch("Vectors must have the same length"))
    s = zero(T)
    for i in eachindex(u, v)
        s += u[i] * v[i]
    end
    return s
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                       Norms:                                                  #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the L1 norm (sum of absolute values) of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for the temporary array from `abs.(v)`
"""
norm_1(v::Vector{T}) where {T} = ∑(abs.(v))

"""
Compute the L2 (Euclidean) norm of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for temporary array from `abs.(v) .^ 2`
"""
norm(v::Vector{T}) where {T} = sqrt(∑(abs.(v) .^ 2))

"""
Compute the infinity norm (maximum absolute value) of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for `abs.(v)`
"""
norm_inf(v::Vector{T}) where {T} = maximum(abs.(v))


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                  Angle between vectors:                                       #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the cosine of the angle between two vectors:

cosθ = u ⋅ v / ( \\|u\\| ⋅ \\|v\\| )

# Complexity
- Time Complexity: O(n) (due to `dot` and `norm` calls)
- Space Complexity: O(n)
"""
angle_cos(u::Vector{T}, v::Vector{T}) where {T} = (u ⋅ v) / (norm(u) * norm(v))



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                  Cross Product:                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the 3D cross product of vectors `u` and `v`.

# Arguments
- `u::Vector{T}`: 3-dimensional vector.
- `v::Vector{T}`: 3-dimensional vector.

# Returns
- A new 3-dimensional vector perpendicular to both `u` and `v`.

# Complexity
- Time Complexity: O(1)
- Space Complexity: O(1)
"""
function cross_product(u::Vector{T}, v::Vector{T}) where {T}
    (size(u) == (3,) && size(v) == (3,)) || throw(ArgumentError("vectors are not 3rd dimensional"))
    a, b, c = u[1], u[2], u[3]
    d, e, f = v[1], v[2], v[3]

    v1 = b * f - c * e
    v2 = -(a * f - c * d)
    v3 = a * e - b * d

    return Vector{T}([v1, v2, v3])
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: multiplication                                      #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #


"""
Multiply a matrix `A` by a vector `v`.

# Complexity
- Let `m = size(A,1)`, `n = size(A,2)`  
- **Time Complexity:** O(m * n)  
- **Space Complexity:** O(m)
"""
function mul(A::Matrix{T}, v::Vector{T}) where {T}
    shape = outputShape(A, v)

    result = Vector{T}(shape)

    for i in 1:shape[1]
        result[i] = ∑(A[i, :] .* v)
    end
    return result
end

"""
Multiply two matrices `A` and `B`.

# Complexity
- Let `A` be m⋅tk, `B` be k⋅n  
- **Time Complexity:** O(m * n * k)  
- **Space Complexity:** O(m * n)
"""
function mul(A::Matrix{T}, B::Matrix{T}) where {T}
    shape = outputShape(A, B)

    result = NDArray{T}(shape)

    for j in 1:shape[2]
        colB = B[:, j]
        for i in 1:shape[1]
            result[i, j] = ∑(A[i, :] .* colB)
        end
    end

    return result
end

# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: trace                                               #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the trace of a square matrix `A` (sum of diagonal elements).

# Arguments
- `A::Matrix{T}`: A square matrix.

# Returns
- Scalar of type `T` representing the sum of the diagonal elements.

# Complexity
- Let `n = size(A, 1)`  
- **Time Complexity:** O(n)  
- **Space Complexity:** O(1)
"""
function trace(A::Matrix{T}) where {T}
    isSquare(A) || throw(ArgumentError("the trace of a matrix exist only if the matrix is square"))

    trace = zero(T)

    for k in 1:size(A, 1)
        trace += A[k, k]
    end

    return trace
end



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: transpose                                           #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the transpose of matrix `A`.

# Returns
- A new matrix with rows and columns swapped.

# Complexity
- Let `m, n = size(A)`  
- **Time Complexity:** O(m * n)  
- **Space Complexity:** O(m * n)
"""
function Base.transpose(A::Matrix)
    A_t = Transpose(A)

    result = similar(A_t)
    copy!(result, A_t)
    return result
end



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: adjoint                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the adjoint (conjugate transpose) of a matrix `A`.

# Returns
- A new matrix where each element is replaced with its complex conjugate and the matrix is transposed.

# Complexity
- Let `m, n = size(A)`  
- **Time Complexity:** O(m * n)  
- **Space Complexity:** O(m * n)
"""
function Base.adjoint(A::Matrix)
    A_t = Adjoint(A)

    result = similar(A)

    for (i, val) in enumerate(A_t)
        result[i] = val
    end
    return result
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: reduced row echelon form                            #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function pivot_selection(R, m, j, pivot_row, tol)

    if (R[pivot_row, j] > tol)
        return pivot_row
    end

    for k in (pivot_row+1):m
        if (abs(R[k, j]) > tol)
            return k
        end
    end

    return pivot_row
end

function row_switch(R::Matrix, pivot_row, pivot_candidate)
    if pivot_candidate != pivot_row
        tmp = copy(R[pivot_row, :])
        R[pivot_row, :] = R[pivot_candidate, :]
        R[pivot_candidate, :] = tmp
    end
end

function eliminate_row(R, i, j, pivot_row, pivot_value=1)
    factor = R[i, j] / pivot_value
    R[i, :] .-= R[pivot_row, :] .* factor
end


"""
Compute the reduced row echelon form (RREF) of a matrix `A`.

# Arguments
- `A::Matrix`: Input matrix.
- `tol::Real`: Tolerance for considering elements as zero (default 1e-10).

# Returns
- A new matrix `R` in RREF.

# Complexity
- Let `m, n = size(A)`  
- **Time Complexity:** O(m * n * min(m, n)) in worst case.  
- **Space Complexity:** O(m * n) (due to copy of matrix)
"""
function reduced_row_echelon_form(A::Matrix; tol::Real=1e-10)

    # We start with the copy of the matrix
    R = copy(A)
    m, n = size(R)
    pivot_row = 1

    # We iterate through the columns
    for j in 1:n

        # If the matrix is rectangular we stop early
        if (pivot_row > m)
            break
        end

        # Pivot Selection: We need to select which row contains the pivot
        pivot_candidate = pivot_selection(R, m, j, pivot_row, tol)

        # Row Switch: We then place the row found below the previous pivot row
        row_switch(R, pivot_row, pivot_candidate)

        # We obtain the value of the pivot
        pivot_value = R[pivot_row, j]

        # If the pivot is not considered 0 then we enter the loop
        if (abs(pivot_value) > tol)

            # We do  normalize the pivot row because it is done in reduced row echelon form
            R[pivot_row, :] ./= pivot_value

            # Elimination: We eliminate all the entries below the pivot
            for i in 1:m
                if (i != pivot_row)
                    eliminate_row(R, i, j, pivot_row)
                end
            end

            # We then move to the next pivot row
            pivot_row += 1
        end
    end

    return R
end


# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: determinant                                         #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

function det2x2(A::AbstractArray{T,2}) where T
    size(A) == (2, 2) || throw(ArgumentError("Matrix must be 2x2"))
    return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
end

function det3x3(A::AbstractArray{T,2}) where T
    size(A) == (3, 3) || throw(ArgumentError("Matrix must be 3x3"))

    isNDArray = A isa NDArray

    det = zero(T)
    for j in 1:3
        # minor = remove first row and column j
        rows = 2:3
        cols = setdiff(1:3, j)
        minor = isNDArray ? A[rows, cols] : @view A[rows, cols]
        println("minor: ", minor, "rows: $rows ", " cols: $cols ", " A: $A" )
        det += (-1)^(1 + j) * A[1, j] * det2x2(minor)
    end
    return det
end

function det4x4(A::AbstractArray{T,2}) where T
    size(A) == (4, 4) || throw(ArgumentError("Matrix must be 4x4"))

    isNDArray = A isa NDArray

    det = zero(T)
    for j in 1:4
        rows = 2:4
        cols = setdiff(1:4, j)
        minor = isNDArray ? A[rows, cols] : @view A[rows, cols]
        det += (-1)^(1 + j) * A[1, j] * det3x3(minor)
    end
    return det
end

"""
Compute the determinant of small matrices (2x2, 3x3, 4x4).

# Returns
- Scalar determinant of type `T`.

# Complexity
- **2x2:** O(1)  
- **3x3:** O(1) (explicit cofactor expansion)  
- **4x4:** O(1)  
- **Space Complexity:** O(1)
"""
function determinant(A::Matrix{T}) where {T}
    shape = size(A)
    if shape == (2, 2)
        return det2x2(A)
    elseif shape == (3, 3)
        return det3x3(A)
    elseif shape == (4, 4)
        return det4x4(A)
    else
        throw(ArgumentError("Handle only up to 4x4 Matrix"))
    end
end



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: inverse                                             #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the inverse of a square matrix using RREF on the augmented matrix [A | I].

# Arguments
- `A::Matrix{T}`: Input square matrix.
- `tol::Real`: Tolerance for numerical zero.

# Returns
- A new matrix representing the inverse of `A`.

# Complexity
- Let `n = size(A, 1)`  
- **Time Complexity:** O(n^3) (due to RREF)  
- **Space Complexity:** O(n^2) (augmented matrix copy)
"""
function inverse(A::Matrix{T}; tol::Real=1e-10) where {T}
    m, n = size(A)

    # 1. Check if the matrix is square
    if m != n
        error("Matrix must be square to compute the inverse.")
    end

    # 2. Create the Augmented Matrix [A | I]
    # 'hcat' concatenates horizontally
    I_n = identityMatrix(n, T) # Identity matrix I of size n
    Aug = hcat(A, I_n)

    # 3. Compute RREF of the Augmented Matrix
    RREF_Aug = reduced_row_echelon_form(Aug; tol=tol)

    # 4. Check for Singularity
    # The left side (columns 1 to n) should now be the Identity Matrix (I).
    # If the matrix is singular, the RREF will have an all-zero row on the left side.

    # Check if the left side of RREF is the identity matrix
    is_identity = true
    for i = 1:n
        # Check if the pivot element on the diagonal is approximately 1
        if abs(RREF_Aug[i, i] - 1.0) > tol
            is_identity = false
            break
        end
    end

    if !is_identity
        error("Matrix is singular and therefore non-invertible.")
    end

    # 5. Extract the Inverse
    # The inverse is the right side (columns n+1 to 2n)
    A_inv = RREF_Aug[:, (n+1):end]

    return A_inv
end



# --------------------------------------------------------------------------------------------- #
#                                                                                               #
#                                   Matrix: rank                                                #
#                                                                                               #
# --------------------------------------------------------------------------------------------- #

"""
Compute the rank of a matrix (number of linearly independent rows).

# Arguments
- `A::Matrix`: Input matrix.
- `tol::Real`: Tolerance for zero elements.

# Returns
- Integer rank of the matrix.

# Complexity
- Let `m, n = size(A)`  
- **Time Complexity:** O(m * n * min(m, n)) (due to RREF)  
- **Space Complexity:** O(m * n)
"""
function rank(A::Matrix; tol::Real=1e-10)
    # 1. Get the Row Echelon Form (R)
    R = reduced_row_echelon_form(A; tol=tol)

    m, n = size(R)
    rank = 0

    # 2. Iterate through rows and count non-zero rows
    for i = 1:m
        # Check if any element in the current row (R[i, :]) is non-zero
        # by seeing if its absolute value is greater than the tolerance.
        is_non_zero_row = any(abs.(R[i, :]) .> tol)

        if is_non_zero_row
            rank += 1
        else
            # Optimization: In REF, if a row is all zeros, all subsequent rows 
            # must also be all zeros, so we can stop counting.
            break
        end
    end

    return rank
end


# broadcast:
# ----------

"""
    broadcast(broadcast_function::Function, arrays::Vararg{AbstractArray})

Applies `broadcast_function` elementwise across multiple arrays with broadcasting support.

# Arguments
- `broadcast_function::Function`: The function to apply elementwise.
- `arrays::AbstractArray...`: One or more arrays to broadcast over.

# Returns
- `NDArray` containing the results of applying `broadcast_function` elementwise to the broadcasted inputs.

# Behavior
- Automatically computes the broadcasted shape of the input arrays.
- Creates a `MultiIter` to iterate over all input arrays simultaneously.
- Applies the function at each broadcasted index.
- Supports arrays of different shapes that are broadcast-compatible.

# Complexity
- Let `N` be the total number of elements in the broadcasted output.
- Let `k` be the number of input arrays.
- Complexity is `O(N * k)` for iteration and function evaluation.
"""
function broadcast(broadcast_function::Function, arrays::Vararg{AbstractArray})
    shape = obtain_broadcast_shape(arrays)

    output_array = NDArray{eltype(first(arrays))}(shape)

    multi_iter = MultiIter(arrays...)

    # Iterate over broadcasted arrays
    for (output_idx, vals) in zip(CartesianIndices(shape), multi_iter)
        output_array[output_idx] = broadcast_function(vals...)
    end
    return output_array
end

function obtain_broadcast_shape(arrays::Tuple{Vararg{Any}})
    padded_shapes = obtain_padded_shapes(arrays)

    maxDim = length(padded_shapes[1])

    output_shape = get_output_shape(padded_shapes, maxDim)

    #@infiltrate
    broadcast_compatible(output_shape, padded_shapes, maxDim) || throw(DomainError("The indices entered are not compatible for broadcasting"))

    return output_shape
end


function obtain_padded_shapes(arrays::Tuple{Vararg{AbstractArray}})
    maxDim = 0
    ref_maxDim = Ref(maxDim)
    return [PaddedShape(array, ref_maxDim) for array in arrays]
end

# the padded shapes must all have the same dimensions
function get_output_shape(padded_shapes::AbstractArray{<:PaddedShape}, maxDim::Int)
    # obtain the dimensions of the first padded shape == max dimension of all original shapes
    return ntuple(i -> maximum(ps[i] for ps in padded_shapes), maxDim)
end

function broadcast_compatible(output_shape::Tuple, padded_shapes::AbstractArray{<:PaddedShape}, maxDim::Int)
    return all(i -> all(ps[i] == 1 || ps[i] == output_shape[i] for ps in padded_shapes), 1:maxDim)
end


# check functions:

function outputShape(u::Vector, v::Vector)
    n = length(u)
    n == length(v) || throw(ArgumentError("the shape of the tensor entered are not compatible"))
    return (n,)
end

function outputShape(A::Matrix, v::Vector)
    m, n1, n2 = size(A, 1), size(A, 2), length(v)
    n1 == n2 || throw(ArgumentError("the shape of the tensor entered are not compatible"))
    return (m,)
end

function outputShape(A::Matrix, B::Matrix)
    m, n1 = size(A, 1), size(A, 2)
    n2, p = size(B, 1), size(B, 2)
    n1 == n2 || throw(ArgumentError("the shape of the tensor entered are not compatible"))
    return (m, p)
end

function check_linear_combination(vectors, coefs, n)
    length(vectors) == length(coefs) || throw(ArgumentError("vectors and coefficients must have the same length"))

    all(length(v) == n for v in vectors) || throw(ArgumentError("all vectors must have the same dimension"))
end