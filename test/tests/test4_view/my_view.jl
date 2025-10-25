using Algebra42
using .TestData
using .TestUtils
using  Base.MultiplicativeInverses: SignedMultiplicativeInverse

function logViewContent(name, data::Bool, write_log_fn)
    write_log_fn("\n" * name)
    write_log_fn(data ? "true" : "false")
end

function logViewContent(name, data, write_log_fn)
    write_log_fn("\n" * name)
    write_content(data, write_log_fn)
end


function test_basic_view2(write_log_fn)
    # -----------------------------
    # Basic contiguous view
    # -----------------------------
   
    A = Algebra42.reshape(collect(0:15), (4, 4))

    println("start test reshaping array: \n\n")
    println("original A: ", A, "strides A: ", strides(A))
    println("test col 1: ", A[1, 1], " ", A[2, 1], " ", A[3, 1], " ", A[4, 1] )
    println("test row 1: ", A[1, 1], " ", A[1,2], " ", A[1, 3], " ", A[1, 4])

    println("test col val: ", A[1], " ", A[2], " ", A[3], " ", A[4])
    println("test row val: ", A[1], " ", A[5], " ", A[9], " ", A[13])
    
  

    println("showing A start: \n\n")
    logViewContent("basic A", A, write_log_fn)

    v = A[2:3, 2:4]
    println("showing V start: \n\n")
    logViewContent("basic view 1", v, write_log_fn)

    logViewContent("basic view size ok", size(v) == (2, 3), write_log_fn)
    logViewContent("basic view access 1 ok", v[1, 1] == A[2, 2], write_log_fn)
    logViewContent("basic view access 2 ok", v[2, 2] == A[3, 3], write_log_fn)

    v[1, 1] = -99

    logViewContent("basic view 1 (modified)", A[2, 2] == -99, write_log_fn)
end


function test_colon_view2(write_log_fn)
    # -----------------------------
    # Colon view
    # -----------------------------
    B = Algebra42.reshape(collect(0:8), (3, 3))
    v_colon = B[:, :]

    logViewContent("colon view 1", v_colon, write_log_fn)

    logViewContent("colon size ok: ", size(v_colon) == size(B), write_log_fn)

    logViewContent("same memory ok: ", pointer(v_colon) == pointer(B), write_log_fn)
end

function test_dimension_droping2(write_log_fn)
    # -----------------------------
    # Dimension dropping
    # -----------------------------
    A = Algebra42.reshape(collect(0:15), (4, 4))

    println("dimension droping start: \n\n\n $A")

    println("row = A[2, :]")

    row = A[ 2, :]

    println("A[2, :] = $(A[2, 1]), $(A[2, 2]), $(A[2, 3]), $(A[2, 4])")

    println("A[2, :] = $(A[2]), $(A[6]), $(A[10]), $(A[14])\n\n")
    
    println("row[2, :] = $(row[1]), $(row[2]), $(row[3]), $(row[4])\n\n")

    logViewContent("row view:", row, write_log_fn)
    
    println("\n\ncol = A[:, 3]")

    col = A[:, 3]

    logViewContent("col view:", col, write_log_fn)

    logViewContent("row view size ok:", size(row) == (4,), write_log_fn)
    logViewContent("col view size ok:", size(col) == (4,), write_log_fn)

    logViewContent("row view access ok:", row[1] == A[2, 1], write_log_fn)
    logViewContent("col view access ok:", col[4] == A[4, 3], write_log_fn)
end

function test_strided_slice2(write_log_fn)
    
    A = Algebra42.reshape(collect(0:15), (4, 4))

    v_stride = A[1:2:4, 1:2:4]

    println("v_stride: ", v_stride)

    logViewContent("strided slice view:", v_stride, write_log_fn)

    logViewContent("strided slice size ok:", size(v_stride) == (2, 2), write_log_fn)

    logViewContent("strided slice access ok 1:", v_stride[1, 1] == A[1, 1], write_log_fn)

    logViewContent("strided slice access ok 2:", v_stride[2, 2] == A[3, 3], write_log_fn)
    
    logViewContent("strided share memory:", pointer(A) == pointer(v_stride), write_log_fn)
end

function test_view_of_view2(write_log_fn)
    A = Algebra42.reshape(collect(0:15), (4, 4))

    v1 = A[2:4, 2:4]

    logViewContent("view of view v1:", v1, write_log_fn)
    
    v2 = v1[2:3, 1:2]
    
    logViewContent("view of view v2:", v2, write_log_fn)

    logViewContent("view v2 size ok:", size(v2) == (2, 2), write_log_fn)

    logViewContent("view v2 access ok:",  v2[1, 1] == A[3, 2], write_log_fn)

    logViewContent("v2 share memory:",  pointer(A) == pointer(v2), write_log_fn)
end

function test_fancy_indexing_copy2(write_log_fn)
    A = Algebra42.reshape(collect(0:15), (4, 4))
    rows = [1, 3, 4]
    cols = [2, 4]
    fancy = A[rows, :][:, cols]  # creates a copy

    logViewContent("fancy index: ", fancy, write_log_fn)

    logViewContent("fancy index is copy: ", pointer(A) != pointer(fancy), write_log_fn)

    logViewContent("fancy index size ok: ", size(fancy) == (3, 2),  write_log_fn)

    logViewContent("fancy index access 1 ok: ", fancy[1, 1] == A[1, 2], write_log_fn)

    logViewContent("fancy index access 2 ok: ", fancy[3, 2] == A[4, 4], write_log_fn)
end

function test_scalar_view2(write_log_fn)
    A = Algebra42.reshape(collect(0:8), (3, 3))

    # -----------------------------
    # Scalar view
    # -----------------------------
    s = A[2, 3]

    logViewContent("scalar view: ", s,  write_log_fn)

    logViewContent("scalar content: ", s == 7,  write_log_fn)

    logViewContent("is numeric: ", typeof(s) <: Number,  write_log_fn)
end

function test_transpose_view2(write_log_fn)
   B = Algebra42.reshape(collect(0:8), (3, 3))
    
    At = transpose(B)

    logViewContent("transpose view: ", At, write_log_fn)
    logViewContent("transpose is view: ", pointer(A) == pointer(At), write_log_fn)
    logViewContent("transpose size ok: ", size(At) == (3, 3), write_log_fn)
    logViewContent("transpose access ok: ", At[1, 2] == B[2, 1], write_log_fn)
end

function test_write_through_view2(write_log_fn)
    B = Algebra42.reshape(collect(0:8), (3, 3))

    v_write = B[2:3, 2:3]

    logViewContent("write through view: ", v_write, write_log_fn)
    v_write[1, 1] = -42

    logViewContent("write through parent: ", B[2, 2] == -42, write_log_fn)
end

function test_trivial_view2(write_log_fn)
    A = Algebra42.reshape(collect(0:8), (3, 3))

    v = A[:, :]
    
    logViewContent("trivial view: ", v, write_log_fn)

    logViewContent("trivial view is same object: ", v === A , write_log_fn)

    logViewContent("trivial view is parent: ",  v === A || (pointer(v) == pointer(A) && strides(v) == strides(A) && size(v) == size(A)), write_log_fn)
end


function test_shape_and_strides_consistency2(write_log_fn)
    # -----------------------------
    # Shape and stride consistency
    # -----------------------------
    C = Algebra42.reshape(collect(0:11), (3, 4))

    v_stride2 = C[1:2:end, :]
    
    logViewContent("v_stride2 view: ", v_stride2, write_log_fn)

    logViewContent("v_stride2 size: ", size(v_stride2) == (2, 4), write_log_fn)

    logViewContent("v_stride2 strides ok: ", stride(v_stride2, 1) == 2 * stride(C, 1), write_log_fn)
end

function test_fancy_indexing_diagonal2(write_log_fn)
    C = Algebra42.reshape(collect(0:15), (4, 4))
    row = NDArray{Int}([[1],[2]])

    v_fancy_diag = C[row, [3,4]]

    logViewContent("v_fancy_diag view: ", v_fancy_diag, write_log_fn)
    logViewContent("v_fancy_diag is copy: ", pointer(C) != pointer(v_fancy_diag), write_log_fn)
    logViewContent("v_fancy_diag size: ", size(v_fancy_diag) == (2,), write_log_fn)
end

function test_nested_view2(write_log_fn)

    A = Algebra42.reshape(collect(0:15), (4, 4))
    # -----------------------------
    # Nested view does not copy
    # -----------------------------
    v_nested1 = A[2:end, :]

    logViewContent("v_nested1 view: ", v_nested1, write_log_fn)


    v_nested2 = v_nested1[:, 3:end]

    logViewContent("v_nested2 view: ", v_nested2, write_log_fn)

    logViewContent("Nested view does not copy view: ", pointer(v_nested2) == pointer(A), write_log_fn)
end

function test_mixed_indexing2(write_log_fn)
    # -----------------------------
    # Mixed indexing for 3D array
    # -----------------------------
    D = Algebra42.reshape(collect(0:26), (3, 3, 3))

    println("start view mixed indexing: \n\n\n$D")


    println("\nv_mixed = D[2, :, 2:3]")

    v_mixed = D[2, :, 2:3]


    logViewContent("v_mixed view: ", v_mixed, write_log_fn)

    logViewContent("v_mixed size ok: ",  size(v_mixed) == (3, 2), write_log_fn)


    logViewContent("v_mixed is not copy: ",  pointer(v_mixed) == pointer(D), write_log_fn)
end

# function test_boolean_mask(write_log_fn)
#     # -----------------------------
#     # Boolean mask creates copy
#     # -----------------------------
#     mask = B .> 4
#     v_mask = B[mask]
#     @assert size(v_mask) == (4,)  # 4 elements greater than 4
#     @assert all(v_mask .> 4)
# end


# -----------------------------------------------------------------------------
# LES TESTS
# -----------------------------------------------------------------------------
struct MockArray{T,N} <: AbstractArray{T,N}
    data::Base.Vector{T}
    dims::Dims{N}
end

MockArray(data::Base.Vector{T}, dims::Dims{N}) where {T,N} = MockArray{T,N}(data, dims)
Base.size(A::MockArray) = A.dims
Base.axes(A::MockArray) = map(Base.OneTo, A.dims)

_checkcontiguous(::Type{Bool}, A::MockArray) = false

function Base.getindex(P::MockArray{T,N}, I::Vararg{Int,N}) where {T,N}
    # Convert Cartesian index I to linear index for the column-major MockArray
    # Assuming N=2: I = (row, col)
    row, col = I[1], I[2]
    rows = size(P)[1]
    
    # Calculate column-major linear index: (col - 1) * rows + row
    lin_idx = (col - 1) * rows + row
    
    return P.data[lin_idx]
end


function test_reshape()

    # --- Préparation des données parentes ---
    # Parent 3x4x5 avec des données séquentielles
    parent_data = Base.reshape(collect(1:60), 3, 4, 5)

    # function test_reshape1(parent_data)
    #     # Tailles du parent: (3, 4, 5). Axes: (1:3, 1:4, 1:5)
    #     parent_axes = axes(parent_data) # (1:3, 1:4, 1:5)
        
    #     # Le calcul de MI est basé sur les tailles des dimensions d'entrée sauf la dernière: (3, 4)
    #     mi = map(SignedMultiplicativeInverse, (3, 4)) 

    #     # Index Linéaire (Vue) -> Index Cartésien (Parent)
        
    #     # Cas 1: Index 1 (Doit mapper à (1, 1, 1))
    #     # remainder_index initial = 0
    #     I1 = _unravel_index(parent_axes, mi, 1)
    #     @assert I1 == (1, 1, 1)
        
    #     # Cas 2: Index 4 (Doit mapper à (1, 2, 1) - Après le premier bloc de 3)
    #     # remainder_index initial = 3
    #     # 1. divrem(3, MI(3)) -> 1, 0. dim_index=0+1=1. dim_remainder=1. MI devient (MI(4),)
    #     # 2. divrem(1, MI(4)) -> 0, 1. dim_index=1+1=2. dim_remainder=0. MI devient ()
    #     # 3. Cas de base: 0 + first(ax[end]) = 5. -> 5.
    #     # ATTENTION: La logique doit mapper à (1, 2, 1) pour l'indexation de Julia.
    #     # L'indexation est basée sur la mémoire. Parent 3x4x5 est colonne-majeur (F-order).
    #     # (1, 1, 1) -> 1
    #     # (2, 1, 1) -> 2
    #     # (3, 1, 1) -> 3
    #     # (1, 2, 1) -> 4
    #     I4 = _unravel_index(parent_axes, mi, 4)
    #     @assert I4 == (1, 2, 1) # (1 + 3 * (2-1) + 3*4*(1-1)) = 4
        
    #     # Cas 3: Index 13 (Doit mapper à (1, 1, 2) - Après le premier 'slice' 3x4=12)
    #     # remainder_index initial = 12
    #     # 1. divrem(12, MI(3)) -> 4, 0. dim_index=1. dim_remainder=4. MI devient (MI(4),)
    #     # 2. divrem(4, MI(4)) -> 1, 0. dim_index=1. dim_remainder=1. MI devient ()
    #     # 3. Cas de base: 1 + first(ax[end]) = 5. -> 5.
    #     I13 = _unravel_index(parent_axes, mi, 13)
    #     @assert I13 == (1, 1, 2) # (1 + 3 * (1-1) + 3*4*(2-1)) = 13
        
    #     # Cas 4: Index 60 (Doit mapper à (3, 4, 5))
    #     I60 = _unravel_index(parent_axes, mi, 60)
    #     @assert I60 == (3, 4, 5)

    #     # Vérification d'un cas intermédiaire
    #     I32 = _unravel_index(parent_axes, mi, 32)
    #     # 32 est l'index 2, 3ème colonne (4-2=2), 3ème couche (3-1=2) -> (2, 3, 3)
    #     # 2 + 3*(3-1) + 12*(3-1) = 2 + 6 + 24 = 32
    #     @assert I32 == (2, 3, 3) 
    # end

    function test_reshape2(parent_data)
        P = parent_data # 3x4x5
        A = Algebra42.reshape(P, (10, 6)) # Vue 10x6
        
        @assert size(A) == (10, 6)
        @assert length(A) == 60
        
        # Test de lecture (lecture de A[i, j] doit donner P[x, y, z])
        # A[1, 1] -> Index Linéaire 1 -> P[1, 1, 1] = 1
        @assert A[1, 1] == 1 
        
        # A[10, 1] -> Index Linéaire 10 -> P[1, 4, 1] (10 = 1 + 3*3 + 12*0)
        # 1 + 3*(4-1) + 12*(1-1) = 10
        @assert A[10, 1] == P[1, 4, 1]
        
        # A[1, 2] -> Index Linéaire 11 -> P[2, 4, 1]
        # 2 + 3*(4-1) + 12*(1-1) = 11
        @assert A[11] == P[2, 4, 1] 
        
        # A[10, 6] -> Index Linéaire 60 -> P[3, 4, 5] = 60
        @assert A[10, 6] == 60 

        # Test d'écriture
        setindex!(A, 999, 5, 3)
        # A[5, 3] est l'index linéaire 25.
        # Index 25 -> P[1, 1, 3] (1 + 3*0 + 12*2 = 25)
        @assert P[1, 1, 3] == 999 
        @assert A[5, 3] == 999
    end

    function test_reshape3()
        # Parent 60 éléments, IndexLinear
        P_linear = collect(101:160) # Vector{Int}
        A_linear = Algebra42.reshape(P_linear, (5, 12)) # Vue 5x12
        
        @assert size(A_linear) == (5, 12)
        @assert A_linear.mi == () # Doit être un ReshapedArray42LF (sans MI)

        # A[1, 1] -> Index Linéaire 1 -> P[1]
        @assert A_linear[1, 1] == 101
        
        # A[5, 1] -> Index Linéaire 5 -> P[5]
        @assert A_linear[5, 1] == 105
        
        # A[1, 2] -> Index Linéaire 6 -> P[6]
        @assert A_linear[1, 2] == 106
        
        # A[5, 12] -> Index Linéaire 60 -> P[60]
        @assert A_linear[5, 12] == 160
        
        # Test d'écriture
        setindex!(A_linear, 777, 3, 7)
        # A[3, 7] est l'index linéaire (3 + 5*(7-1)) = 33.
        @assert P_linear[33] == 777
        @assert A_linear[3, 7] == 777
    end

    function test_reshape4()
        # 1. Define a parent array P (simulating a 3x4 array)
        P = MockArray(collect(1:12), (3, 4)) # Parent is 3x4

        # 2. Calculate the mi field
        # Dimensions before the last one: (3,)
        # mi should contain SignedMultiplicativeInverse(3)
        parent_dims_no_last = (3,) 
        mi_tuple = map(SignedMultiplicativeInverse, parent_dims_no_last)

        # 3. Create a ReshapedArray42 (e.g., reshape 3x4 into a 12-element Vector)
        R = ReshapedArray42(P, (12,), mi_tuple)

       # println("--- ReshapedArray42 with Multiplicative Inverse (MI) Test ---")
       # println("Parent (3x4): ", P.data, " -> Index Cartesian (not Linear)")
        #println("Reshaped View (12,): ", size(R))
       # println("MI field: ", R.mi) # R.mi is non-empty, triggering the conversion logic

        # Test Access: R[7]
        # R[7] is the 7th element linearly. In P (3x4, column-major):
        # Col 1: 1, 2, 3
        # Col 2: 4, 5, 6
        # Col 3: 7, 8, 9 <-- R[7] should map to P[1, 3] which holds value 7
        # The original P[1, 3] (value 7) should be accessed.

        # Accessing R[7] triggers: 
        #   _sub2ind( (12,), 7 ) -> 7 
        #   ind2sub_rs( (1:3, 1:4), (SignedMultiplicativeInverse(3),), 7 ) -> (1, 3) 
        #   parent(R)[1, 3] -> 7

        result = getindex(R, 7)
        @assert result == 7
       # println("\nAccessing R[7]:")
        #println("  View linear index: 7")
        #println("  Parent Cartesian index (calculated using MI logic): (1, 3)")
       # println("  Value: ", result)

        # R[10] should map to P[1, 4] which holds value 10
        result_2 = getindex(R, 10)

        @assert result_2 == 10
       # println("\nAccessing R[10]:")
        #println("  View linear index: 10")
       # println("  Parent Cartesian index (calculated using MI logic): (1, 4)")
       # println("  Value: ", result_2)

    end

    try
       # test_reshape1(parent_data)
        test_reshape2(parent_data)
        test_reshape3()
        test_reshape4()
    catch e
        if (e isa AssertionError)
            write_log_fn("test_reshape: failed")
        else
            rethrow(e)
        end
    end
end


function run_tests_my_view(write_log_fn)
    test_basic_view2(write_log_fn)
    test_colon_view2(write_log_fn)
    test_dimension_droping2(write_log_fn) # scalar
    test_strided_slice2(write_log_fn)
    test_view_of_view2(write_log_fn)
    test_fancy_indexing_copy2(write_log_fn)
    test_scalar_view2(write_log_fn)
    #test_transpose_view2(write_log_fn)
    test_write_through_view2(write_log_fn)
    test_trivial_view2(write_log_fn)
    test_shape_and_strides_consistency2(write_log_fn)
    test_fancy_indexing_diagonal2(write_log_fn)
    test_nested_view2(write_log_fn)
    test_mixed_indexing2(write_log_fn)
    test_reshape()
   # test_boolean_mask(write_log_fn)
end