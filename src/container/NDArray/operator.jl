import Base.Broadcast: BroadcastStyle, combine_styles, materialize, copy!, Broadcasted


# The content of this file permits to use the broadcast syntax suxh as v .+= λ .* v

# We create a new broadcast style and we assign it to our type
struct NDArrayStyle <: BroadcastStyle end

Base.Broadcast.BroadcastStyle(::Type{<:NDArray}) = NDArrayStyle()

Base.similar(bc::Broadcasted{NDArrayStyle}, ::Type{ElType}) where {ElType} = begin
    println("Called NDArray similar!")

   # T = Base.Broadcast.broadcasted_eltype(bc)
    axes_bc = Base.Broadcast.axes(bc)
    dims = tuple(length.(axes_bc)...) 

    dest = NDArray{ElType}(dims)
    return dest
end

# Copy data from a Broadcasted object to a destination MyVector
function Base.Broadcast.copy!(dest::NDArray, bc::Broadcast.Broadcasted{NDArrayStyle})
    println("Called copy")
    for (i, val) in enumerate(bc)
        dest[i] = val
    end
    return dest
end


Base.BroadcastStyle(::NDArrayStyle, ::NDArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::NDArrayStyle, ::Base.Broadcast.DefaultArrayStyle) = NDArrayStyle()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle, ::NDArrayStyle) = NDArrayStyle()

import Base: *, +, -



# --- Addition ---
+(a::NDArray, b::NDArray) = begin
    println("Called +(NDArray, NDArray)")
    add(a, b)
end

+(a::Number, b::NDArray) = begin
    println("Called +(Number, NDArray) -> forwards to +(NDArray, Number)")
    +(b, a)
end

+(a::NDArray, b::Number) = begin
    println("Called +(NDArray, Number)")
    add(a, b)
end


# --- Subtraction ---
-(a::NDArray, b::NDArray) = begin
    println("Called -(NDArray, NDArray)")
    sub(a, b)
end

-(a::Number, b::NDArray) = begin
    println("Called -(Number, NDArray) -> forwards to -(NDArray, Number)")
    -(b, a)
end

-(a::NDArray, b::Number) = begin
    println("Called -(NDArray, Number)")
    sub(a, b)
end


# --- Multiplication ---
*(a::NDArray, b::Number) = begin
    println("Called *(NDArray, Number)")
    prod(a, b)
end

*(a::Number, b::NDArray) = begin
    println("Called *(Number, NDArray) -> forwards to *(NDArray, Number)")
    *(b, a)
end

# --- Multiplication ---
*(a::NDArray, b::Number) = begin
    println("Called *(NDArray, Number)")
    prod(a, b)
end






