# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                               #
#                                               Shape:                                          #
#                                                                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# Wrapper around the dimension representing the shape, also used to find the dimensions of a nested array for example

mutable struct Shape{N}
    length::Int
    dims::NTuple{N, Int}
    
    # Inner constructor for scalars
    Shape{0}() = new{0}(1, ())

    # Inner constructor for a general Shape from a tuple of dimensions
    function Shape{N}(tuple::NTuple{N, Int}) where {N}
        if any(d -> d < 0, tuple)
            throw(DomainError("Dimensions cannot be negative"))
        end
        new{N}(prod(tuple), tuple)
    end
end

# ======== Constructors ============================================= #

# scalar
Shape(scalar::Number) = Shape{0}()

# from tuple
Shape(tuple::NTuple{N, Int})  where {N} = Shape{N}(tuple)

# from NestedArray
function Shape(array::AbstractArray{T, 1}; dtype = Float32) where T 
 if (isempty(array))
        return Shape((0,));
    elseif (dtype == Any) # ragged array
        return Shape((length(array),));
    end
    return Shape(getShape(array))
end

# ======== Core properties ============================================= #

Base.ndims(shape::Shape{N}) where {N} = N

Base.length(shape::Shape{N}) where {N} = shape.length

Base.size(shape::Shape{N}) where {N} = shape.dims

# ======== Iteration ================================================== #

Base.iterate(shape::Shape) = iterate(shape.dims)

Base.iterate(shape::Shape, state) = iterate(shape.dims, state)

# ========  Determines recursively the shape of a nested array ========== #

function getShape(array)
    # The input can be eiter a collection or a single element
    
    # If it is a collection
    if (isa(array, AbstractVector))
        # We then obtain the tupple representing the dimensions of each entries
        dimSubArray = map(getShape, array)
        # We have the following possibilites:
        
        # 1. It the actual array is a vector ( 1D array ), in this case all entry of dimSubArray
        #    are nothing, this also mean that a single tuple containing a single value must be returned
        if all(x -> isnothing(x), dimSubArray)
            return (length(array),);
        # 2. The actual array is n dimensional array, but all entries have the same shape, in this case all tupple stored
        #    stored in dimSubArray are equal
        elseif all(x -> x == dimSubArray[1], dimSubArray)
            return (length(array), dimSubArray[1]...);
        # 3. The actual array is n dimrndionsl array but not all entries are the same this mean that it is a jagged array
        #    however a jagged array can be created only if dtype
        else
            throw(DomainError("Verify Error"))
        end
    # If it is a single element    
    else
        # In this case we return nothing to tell that the last call of this function must
        # create a tupple of only one element, that is the length of the previous array
        return nothing;
    end
end
