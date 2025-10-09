"Simple prod function that compute the product of all the elements inside an array"

function add(elem1::Number, elem2::Number)
     println("Called +(elem1::Number, elem2::Number)")
    return elem1 + elem2;
end

function sub(elem1::Number, elem2::Number)
     println("Called -(elem1::Number, elem2::Number)")
    return elem1 - elem2;
end

function prod(elem1::Number, elem2::Number)
     println("Called *(elem1::Number, elem2::Number)")
    return elem1 * elem2;
end

function prod(array)
    product = (isempty(array) || isnothing(array)) ? 0 : 1
    for element in array
        product *= element
    end
    return product
end

function sum(array)
    isempty(array) && throw(ArgumentError("There is no element to sum"))
    sum = 0
    for element in array
        sum += element
    end
    return sum
end


function abs(nb::Number)
    println("called ????")
    return nb < 0 ? -nb : nb
end