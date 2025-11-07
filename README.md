# Algebra42.jl

**Algebra42.jl** is a **comprehensive linear algebra library in Julia** that provides a full suite of operations for vectors, matrices, and tensors. Designed for numerical computations, educational purposes, and mathematical experimentation, it implements many common linear algebra functions from scratch with intuitive syntax and custom error handling.

---

## Features

### 1. Vector Operations
- **Linear combination**: `linear_combination(vectors, coefs)` – combines vectors with given coefficients.  
- **Interpolation (LERP)**: `lerp(a, b, t)` – linear interpolation for scalars or vectors.  
- **Dot product**: `u ⋅ v` or `dot(u, v)`.  
- **Cross product**: `cross_product(u, v)` – only for 3D vectors.  
- **Vector norms**:
  - `norm_1(v)` – L1 norm (sum of absolute values).  
  - `norm(v)` – L2 norm (Euclidean).  
  - `norm_inf(v)` – Infinity norm (max absolute value).  
- **Angle between vectors**: `angle_cos(u, v)` – cosine of the angle between vectors.

### 2. Tensor Arithmetic
Supports element-wise operations with broadcasting:
- `add(a, b)` – addition of tensors or tensor and scalar.  
- `sub(a, b)` – subtraction of tensors or tensor and scalar.  
- `prod(a, scalar)` – multiply tensor by a scalar.

### 3. Matrix Operations
- **Matrix multiplication**: `mul(A, v)` for matrix-vector, `mul(A, B)` for matrix-matrix.  
- **Trace**: `trace(A)` – sum of diagonal elements.  
- **Transpose**: `transpose(A)` – returns the transposed matrix.  
- **Determinant**: `determinant(A)` – supports 1×1 to 4×4 directly and recursively for larger matrices.  
- **Inverse**: `inverse(A)` – computes the inverse using augmented RREF.  
- **Rank**: `rank(A)` – computes matrix rank.  
- **Reduced Row Echelon Form (RREF)**: `reduced_row_echelon_form(A)` – Gaussian elimination based.

### 4. Advanced Indexing, Views, and Custom Broadcasts
Algebra42.jl implements a **custom broadcasting system**, advanced indexing, slicing, and multiple views for tensors:
- **Custom broadcasting**: `@MyBroadcast` allows element-wise operations with optimized performance.  
- **Advanced slicing**: Supports selecting subarrays, rows, columns, and multidimensional slices.  
- **Multiple views**: Efficiently reference submatrices or subtensors without copying data.  
- These features allow expressive, high-performance computations like NumPy or Julia’s built-in arrays while retaining full control.

### 5. Utility Functions
- `isSquare(A)` – checks if a matrix is square.  
- `outputShape(...)` – computes expected output shapes for multiplication.  
- `IdentityMatrix(n, T)` – generates an identity matrix of size n×n.  
- Custom error messages for dimension mismatch, singular matrices, and non-square matrices.

### 6. Error Handling
Algebra42.jl provides robust error checking:
- Throws `ArgumentError` for dimension mismatches or non-square matrices.  
- Detects singular matrices when computing inverses.  
- Tolerance-based numerical checks to handle floating-point inaccuracies.

---

## Example Usage

```julia
using Algebra42

# Vector operations
v1 = [1, 2, 3]
v2 = [4, 5, 6]

println("Dot product: ", v1 ⋅ v2)
println("Cross product: ", cross_product(v1, v2))
println("Linear combination: ", linear_combination([v1, v2], [0.5, 2]))
println("L2 norm: ", norm(v1))

# Matrix operations
A = [1 2; 3 4]
B = [5 6; 7 8]

println("Matrix multiplication: ", mul(A, B))
println("Determinant of A: ", determinant(A))
println("Inverse of A: ", inverse(A))
println("Rank of A: ", rank(A))

# Reduced row echelon form
C = [1 2 3; 4 5 6; 7 8 9]
println("RREF of C: ", reduced_row_echelon_form(C))

# Custom broadcasting
D = [1 2 3]
E = 2
@MyBroadcast F = D .* E .+ 1
println("Custom broadcast result: ", F)
