

# "Simple sum function that compute the total value of the elements inside an array" 
# function sum(array::AbstractArray{T})::Int where T
#     total::Int = 0;
#     for element in array
#         total += element;
#     end
#     return total;
# end

"Simple prod function that compute the product of all the elements inside an array" 
function prod(array)::Int
    product::Int = isempty(array) || isnothing(array) ? 0 : 1 ;
    for element in array
        product *= element;
    end
    return product;
end

