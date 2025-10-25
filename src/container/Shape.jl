const Collection = Base.Vector



"""
    Shape{N}

Represents the dimensions of a multi-dimensional object, with `N` being the
number of dimensions.

# Fields
- `size::Int`: The total number of elements in the shape.
- `shape::NTuple{N, Int}`: A tuple representing the size of each dimension.
"""
mutable struct Shape{N}
    length::Int          # size
    dims::NTuple{N, Int} # shape
    
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


"""
    ndims(shape::Shape{N})

Returns the number of dimensions of a `Shape` object.

# Arguments
- `shape::Shape{N}`: The Shape object.

# Returns
- `Int`: The number of dimensions `N`.
"""
Base.ndims(shape::Shape{N}) where {N} = N

Base.length(shape::Shape{N}) where {N} = shape.length

Base.size(shape::Shape{N}) where {N} = shape.dims

# Outer constructors for Shape

"""
    Shape(x::Number)

Creates a `Shape` from a scalar value. A 0-dimensional shape
always has a size of 1.

# Arguments
- `x::Number`: The scalar number.

# Returns
- `Shape{0}`: A 0-dimensional Shape object.
"""
Shape(scalar::Number) = Shape{0}()


"""
    Shape(tuple::NTuple{N, Int})

Creates an N-dimensional `Shape` from a tuple of integers.

# Arguments
- `tuple::NTuple{N, Int}`: A tuple where each element is the size of a dimension.

# Returns
- `Shape{N}`: A Shape object with the corresponding number of dimensions.
"""
Shape(tuple::NTuple{N, Int})  where {N} = Shape{N}(tuple)



"""
    Shape(array::Collection{Any}; dtype = Float32)

Creates a `Shape` from a collection of data, handling empty and nested arrays.

This constructor infers the shape of the array based on its contents. It
distinguishes between empty arrays, arrays of `Any` type, and arrays with
a consistent nested structure.

# Arguments
- `array::Collection{Any}`: The input collection.
- `dtype = Float32`: An optional data type to guide shape inference,
  especially for collections of `Any`.

# Returns
- `Shape`: The inferred Shape of the array.

# Throws
- `DomainError`: If the nested arrays are "jagged" (i.e., do not have a
  consistent shape).
"""
function Shape(array::Collection; dtype = Float32) 
 if (isempty(array))
        return Shape((0,));
    elseif (dtype == Any)
        return Shape((length(array),));
    end
    return Shape(getShape(array))
end


"""
    getShape(array)

Recursively determines the shape of a potentially nested array.

This is a helper function that traverses a collection to infer its full
dimensions. It handles two main cases: collections and single elements.

# Arguments
- `array`: The input collection or single element.

# Returns
- `Union{Tuple{Vararg{Int}}, Nothing}`: A tuple of integers representing the
  shape, or `nothing` if the input is a single element.

# Throws
- `DomainError`: If the array is a jagged array, where nested collections
  have inconsistent shapes.
"""
function getShape(array)
    # The input can be eiter a collection or a single element
    
    # If it is a collection
    if (isa(array, Collection))
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


function getShapeBottomUp(array)
    # The input can be eiter a collection or a single element
    
    # If it is a collection
    if (isa(array, Collection))
        # Get shapes of each element
        dimSubArray = map(getShapeBottomUp, array)

        # If all elements are nothing, we reached the innermost level
        if all(isnothing, dimSubArray)
            return (length(array),);
        elseif all(x -> x == dimSubArray[1], dimSubArray)
            # Prepend current length to the shape of the inner array
            return (dimSubArray[1]..., length(array))
        else
            throw(DomainError("Jagged array"))
        end
    # If it is a single element    
    else
        # In this case we return nothing to tell that the last call of this function must
        # create a tupple of only one element, that is the length of the previous array
        return nothing;
    end
end



# Iteration
function Base.iterate(shape::Shape)
    iterate(1:ndims(shape))
end

function Base.iterate(shape::Shape, state)
    state > length(shape.dims) &&  return nothing 
    return shape.dims[state], state + 1
end