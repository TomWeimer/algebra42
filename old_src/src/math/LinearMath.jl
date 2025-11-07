using .Errors: ERR_IS_NOT_SQUARE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                       Linear Math:                                            #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# ======== Math Symbols ==================================================== #

# Sum
∑(arrayToSum) = sum(arrayToSum)

# dot product
⋅(u::Vector{T}, v::Vector{T}) where {T} = dot(u, v)

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       tensor arithmetics:                                     #
#_______________________________________________________________________________________________#

# +
add(a::AbstractNDArray, b::AbstractNDArray) = @MyBroadcast a .+ b
add(a::AbstractNDArray, scalar::Number)     = @MyBroadcast a .+ scalar

# -
sub(a::AbstractNDArray, b::AbstractNDArray) =  @MyBroadcast a .- b
sub(a::AbstractNDArray, scalar::Number)     =  @MyBroadcast a .- scalar

# *
prod(a::AbstractNDArray, scalar::Number)    = @MyBroadcast a .* scalar

#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾#
#                                       linear operations:                                     #
#_______________________________________________________________________________________________#


# ======== linear combination ==================================================== #

function check_linear_combination(vectors, coefs)
    length(vectors) == length(coefs) || throw(ArgumentError("vectors and coefficients must match"))
    all(length(v) == length(vectors[1]) for v in vectors) || throw(ArgumentError("all vectors must have same dimension"))
end

function linear_combination(vectors::AbstractVector{Vector{T}}, coefs::AbstractVector{T}) where {T}
    n = length(vectors[1])

    check_linear_combination(vectors, coefs)

    result = zeroVector((n,), T)

    @MyBroadcast begin
        for (v, λ) in zip(vectors, coefs)
            result .+= λ .* v
        end
    end

    return result
end

# ======== linear interpolation ==================================================== #

# scalar
function lerp(a::Number, b::Number, t::Number)
    return ((1 - t) * a) + (t * b)
end

# vectors
function lerp(u::V, v::V, t::Number) where {V<:AbstractArray}
    A =  @MyBroadcast (1 - t) .* u .+ t .* v
    println("LERP type is: ", typeof(A))
    println("LERP  is: ", A)
    return A
end


# ======== dot product ==================================================== #

function dot(u::Vector{T}, v::Vector{T}) where {T}
    length(u) == length(v) || throw(DimensionMismatch("Vectors must have the same length"))
    s = zero(T)
    for i in eachindex(u, v)
        s += u[i] * v[i]
    end
    return s
end

# ======== norms  ======================================================== #

"""
Compute the L1 norm (sum of absolute values) of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for the temporary array from `abs.(v)`
"""
norm_1(v::Vector{T}) where {T} = @MyBroadcast ∑(abs.(v))

"""
Compute the L2 (Euclidean) norm of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for temporary array from `abs.(v) .^ 2`
"""
norm(v::Vector{T}) where {T} = @MyBroadcast sqrt(∑(abs.(v) .^ 2))

"""
Compute the infinity norm (maximum absolute value) of vector `v`.

# Complexity
- Time Complexity: O(n)
- Space Complexity: O(n) for `abs.(v)`
"""
norm_inf(v::Vector{T}) where {T} = @MyBroadcast maximum(abs.(v))


# ======== angle between vectors  ======================================================== #

"""
Compute the cosine of the angle between two vectors:

cosθ = u ⋅ v / ( \\|u\\| ⋅ \\|v\\| )

# Complexity
- Time Complexity: O(n) (due to `dot` and `norm` calls)
- Space Complexity: O(n)
"""
angle_cos(u::Vector{T}, v::Vector{T}) where {T} = (u ⋅ v) / (norm(u) * norm(v))



# ======== cross product (3D)  ================================================================ #

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

# ======== matrix: multplication  ================================================================ #

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

    @MyBroadcast begin
        for i in 1:shape[1]
            result[i] = ∑(A[i, :] .* v)
        end
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

    @MyBroadcast begin
        for j in 1:shape[2]
            colB = B[:, j]
            for i in 1:shape[1]
                result[i, j] = ∑(A[i, :] .* colB)
            end
        end
    end

    return result
end

# ======== matrix: multplication  ================================================================ #

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

# ======== matrix: transpose  ================================================================ #

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
    println("doing transpose\n\n of $A")
    A_t = Transpose(A)
    println("transpose: $A_t")
    println("size: ", size(A_t))
    println("original: ", A)
    println("axes: $(axes(A_t)). axes original: $(axes(A))")
    return copy(A_t)
end


# ======== matrix: reduced row echelon form  ================================================================ #

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
    @MyBroadcast R[i, :] .-= R[pivot_row, :] .* factor
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
            @MyBroadcast R[pivot_row, :] ./= pivot_value

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


# ======== matrix: determinant  ================================================================ #

function det2x2(A::AbstractMatrix{T}) where T
    size(A) == (2, 2) || throw(ArgumentError("Matrix must be 2x2"))
    return A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
end

function det3x3(A::AbstractMatrix{T}) where T
    size(A) == (3, 3) || throw(ArgumentError("Matrix must be 3x3"))

    det = zero(T)
    for j in 1:3
        rows = 2:3
        cols = setdiff(1:3, j)
        minor = @view A[rows, cols]
        det += (-1)^(1 + j) * A[1, j] * det2x2(minor)
    end
    return det
end

function det4x4(A::AbstractMatrix{T}) where T
    size(A) == (4, 4) || throw(ArgumentError("Matrix must be 4x4"))

    det = zero(T)
    for j in 1:4
        rows = 2:4
        cols = setdiff(1:4, j)
        minor = @view A[rows, cols]
        det += (-1)^(1 + j) * A[1, j] * det3x3(minor)
    end
    return det
end

function determinant(A::AbstractMatrix{T}) where T
    n, m = size(A)
    n == m || throw(ArgumentError("Matrix must be square"))

    if n == 1
        return A[1,1]
    elseif n == 2
        return det2x2(A)
    elseif n == 3
        return det3x3(A)
    elseif n == 4
        return det4x4(A)
    else
        det = zero(T)
        for j in 1:n
            rows = 2:n
            cols = setdiff(1:n, j)
            minor = @view A[rows, cols]
            det += (-1)^(1 + j) * A[1,j] * determinant(minor)
        end
        return det
    end
end


# ======== matrix: inverse  ================================================================ #

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
    n = size(A, 1)

    # 1. Check if the matrix is square
    isSquare(A) || @throw_error ArgumentError ERR_IS_NOT_SQUARE

    # 2. Create the Augmented Matrix [A | I_n]
    I_n = IdentityMatrix(n, T)
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



# ======== matrix: rank  ================================================================ #

function rank(A::Matrix; tol::Real=1e-10)
    # 1. Get the Row Echelon Form (R)
    R = reduced_row_echelon_form(A; tol=tol)

    m, n = size(R)
    rank = 0

    # 2. Iterate through rows and count non-zero rows
    for i = 1:m
        # Check if any element in the current row (R[i, :]) is non-zero
        # by seeing if its absolute value is greater than the tolerance.
        @MyBroadcast is_non_zero_row = any(abs.(R[i, :]) .> tol)

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
