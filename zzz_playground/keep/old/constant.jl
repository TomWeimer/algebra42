
# Return a single elements of the array ( if enough are provided )
const Index = Int                          

# Return a view if:
# - a colon is present
# - the number of indices given is less than the number of array's dimensions
const ViewIndices = Union{AbstractRange, Int, Colon, Base.Slice} 

# Fancy indices are arrays of integer or boolean
const FancyIndices = Union{AbstractArray{Int},AbstractArray{Bool}}

# Return a copy of the array, if at least one fancy index is present
const CopyIndices = Union{AbstractRange, Int, Colon, Base.Slice, AbstractArray{Int}, AbstractArray{Bool}}



# abstract type AbstractNDArray{T, N} <: AbstractArray{T, N} end


