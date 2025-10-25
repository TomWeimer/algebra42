"Simple prod function that compute the product of all the elements inside an array"

function add(elem1::Number, elem2::Number)
    return elem1 + elem2;
end

function sub(elem1::Number, elem2::Number)
    return elem1 - elem2;
end

function prod(elem1::Number, elem2::Number)
    return elem1 * elem2;
end

function prod(array)
    product = (isempty(array) || isnothing(array)) ? 0 : 1
    for element in array
        product *= element
    end
    return product
end

# Can be used with a lazy generator
function sum(iter)
    state = iterate(iter)
    state === nothing && return 0  # empty iterable

    x, st = state
    s = zero(typeof(x))            # use type of first element
    s += x

    while true
        next_state = iterate(iter, st)
        next_state === nothing && break
        x, st = next_state
        s += x
    end

    return s
end


function abs(nb::Number)
    return nb < 0 ? -nb : nb
end