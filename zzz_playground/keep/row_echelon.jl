
# function row_echelon(A::Matrix; tol::Real=1e-10)
#     n = size(A)[1]
    
#     # 1. Initialize
#     R = copy(A) # Work on a mutable floating-point copy
#     m, n = size(R)
#     pivot_row = 1


#     println("matrix:  R= $(R) \n")
    
#     # 2. Iterate through columns (j)
#     for j = 1:n
#         # If we run out of rows, stop
#         privot_row()
#         if pivot_row > m
#             break
#         end

#         println("\n matrix after switch:  R= $(R)")

#         # --- Elimination ---
#         # Check if the pivot element is non-zero (above tolerance)
#         pivot_value = R[pivot_row, j]
#         if abs(pivot_value) > tol
            
#             # 1. Normalize the pivot row (make the pivot element equal to 1)
#             # This is optional for REF but required for RREF. We do it here for clarity.
#             R[pivot_row, :] ./= pivot_value

#         # 2. Eliminate all entries below the pivot

#             println("Enter loop: ", pivot_row + 1, " m: ", m)
#             for i in (pivot_row + 1):m

                
#                 factor = R[i, j] # The multiple of the pivot row to subtract
                
#                 # Perform the row operation: R[i, :] = R[i, :] - factor * R[pivot_row, :]
#                 # Use @views for efficiency and prevent unnecessary memory allocation

#                 println("to change: R[$i, :]: $(R[i, :]) using R[$pivot_row, :] = $(R[pivot_row, :])")
#                 println("operation done:  R[$i, :] - $factor * R[$pivot_row] = $(R[i, :]) -  $factor * $(R[pivot_row, :]) ")
                
#                 R[i, :] .-= factor .* R[pivot_row, :]

#                 println("to Insert:  R[$i, :] $(R[i, :]) ")
                
#             end

#              println("\nresult:  R =  $(R) ")

#             # Move to the next pivot row
#             pivot_row += 1
#         end
       
#         # If the pivot is (numerically) zero, we skip this column and look for a pivot in the next column.
#     end
#     println("Given: ", R)

#     return R
# end



#function row_echelon(A::Matrix; tol::Real=1e-10)
 #   return reduced_row_echelon_form(A, tol=tol)
    # R = copy(A)
    # m, n = size(R)
    # pivot_row = 1

    # # We start with the copy of the matrix
    # println("\nrow echelon form of: $A")


    # # We iterate through the columns
    # for j in 1:n

    #     # If the matrix is rectangular we stop early
    #     if (pivot_row > m)
    #         break
    #     end

    #     println(" pivot row: ", R[pivot_row, :])

    #     # Pivot Selection: We need to select which row contains the pivot
    #     # pivot_candidate = pivot_selection(R, m, j, pivot_row)

    #     # Row Switch: We then place the row found below the previous pivot row
    #     # row_switch(R, pivot_row, pivot_candidate)

    #     # We obtain the value of the pivot
    #     pivot_value = R[pivot_row, j]

    #     # If the pivot is not considered 0 then we enter the loop
    #     if (abs(pivot_value) > tol)

    #         # We do not normalize the pivot row because it is something that must be done in reduced row echelon form

    #         # Elimination: We eliminate all the entries below the pivot
    #          for i in (pivot_row + 1):m
    #             eliminate_row_below_pivot(R, i, j, pivot_row, pivot_value)              
    #         end

    #         # We then move to the next pivot row
    #         pivot_row += 1
    #     end

    #     println(" result obtained after op: $R\n")
    # end

    # println("row echelon form obtained: $R\n")
    # return R
#end



# function pivot_selection(R, m, j, pivot_row, tol)

#     if (R[pivot_row, j] > tol)
#         return pivot_row
#     end

#     for k in (pivot_row + 1):m
#         if (abs(R[k, j]) > tol)
#            return k
#         end
#     end
    
#     return pivot_row
# end

# function row_switch(R::Matrix, pivot_row, pivot_candidate)
#     if pivot_candidate != pivot_row
#         println("Operation done: R$pivot_row <-> R$pivot_candidate")
#         println("               before:")
#         println("               $(R[pivot_row, :]) <-> $(R[pivot_candidate, :])")

#         tmp = copy(R[pivot_row, :])
#         R[pivot_row, :] = R[pivot_candidate, :]
#         R[pivot_candidate, :] = tmp

#         println("               after:")
#         println("               $(R[pivot_row, :]) <-> $(R[pivot_candidate, :])")
#     end
# end

# function eliminate_row(R, i, j, pivot_row, pivot_value = 1)
 
#     entry_to_eliminate = R[i, j]
    
#     # To elimate the row below the pivot, we want to obtain 0 to the entry below the pivot, this done as:
#     #               row_to_eliminate -= pivot_row * (entry_to_eliminate / pivot_value )

#     factor = entry_to_eliminate / pivot_value

#     println("Operation done: R$i = R$i - R$pivot_row * $factor")
#     println("                R$i = $(R[i, :]) - $(R[pivot_row, :]) * $factor")
    
#     R[i, :] .-= R[pivot_row, :] .* factor

#     println("                    = $(R[i, :])")
# end


# function reduced_row_echelon_form(A::Matrix; tol::Real=1e-10)
#     R = copy(A)
#     m, n = size(R)
#     pivot_row = 1

#     # We start with the copy of the matrix
#     println("\nrow echelon form of: $A")


#     # We iterate through the columns
#     for j in 1:n

#         # If the matrix is rectangular we stop early
#         if (pivot_row > m)
#             break
#         end

#         println(" pivot row: ", R[pivot_row, :])

#         # Pivot Selection: We need to select which row contains the pivot
#         pivot_candidate = pivot_selection(R, m, j, pivot_row, tol)

#         # Row Switch: We then place the row found below the previous pivot row
#         row_switch(R, pivot_row, pivot_candidate)

#         # We obtain the value of the pivot
#         pivot_value = R[pivot_row, j]

#         # If the pivot is not considered 0 then we enter the loop
#         if (abs(pivot_value) > tol)

#             # We do  normalize the pivot row because it is done in reduced row echelon form
#             R[pivot_row, :] ./= pivot_value

#             # Elimination: We eliminate all the entries below the pivot
#              for i in 1:m
#                 if (i != pivot_row)
#                     eliminate_row(R, i, j, pivot_row)
#                 end
#             end

#             # We then move to the next pivot row
#             pivot_row += 1
#         end

#         println(" result obtained after op: $R\n")
#     end

#     println("row echelon form obtained: $R\n")
#     return R
# end