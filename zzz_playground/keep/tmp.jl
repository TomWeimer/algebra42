# works only in scalar
function Base.getindex(array::NDArray{DType, 0}, args::Tuple{}) where {DType}
   isempty(args) || throw(DomainError("Not working on scalar"))
   return item(array);
end

function Base.getindex(::NDArray{DType, N}, args::Tuple{Int}) where {DType, N}
   # Check that the tuple is not empty
   isempty(args) &&  throw(DomainError("The dimension is not 0 ndarray[()] works only on scalar"))
   # Throw an error the element can not be accessed with a tuple
   throw(DomainError("The element cannnot be accessed with tuple"))
end

# ndarray[1] not working in scalar
function Base.getindex(array::NDArray{DType, 0}, indices::Vararg{Union{Int, Colon}} ) where {DType}
   throw(DomainError("accessing by index with a scalar is only possible as ndarray[()] or ndarray.item()"))
end

# ndarray[1, 2, 3]
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Int} ) where {DType, N}
   #@infiltrate
   # Check that the indices are not empty
   isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
   # Check if the index are valid

   dim = ndims(array);
   indicesNb = length(indices)

   (indicesNb <= dim) ||  throw(DomainError("Indexes out of bounds"))
   
   indicesStretched = indices


   (elementsAreValid(size(array), indices...)) || throw(DomainError("Indexes out of bounds"))

   if (indicesNb < dim)
      indicesStretched = ntuple(i -> i <= indicesNb ? indices[i] : Colon(), dim)
   end

   # If the index is valid return the associated elements
   return array.content[_offset(array, indicesStretched...)];
end

# ndarray[1, :, 3]
function Base.getindex(array::NDArray{DType, N}, indices::Vararg{Union{Int, Colon}} ) where {DType, N}
   # Check that the indices are not empty
   isempty(indices) && throw(DomainError("Cannot access to element with ndarray[]"))
   # Check if the index are valid
   #@infiltrate
   _isValidIndex(array.shape, indices...) || throw(DomainError("Indexes out of bounds"))
   # If the index is valid return the associated elements
   return @view array[indices...];
end

# using cartesian index
function Base.getindex(A::NDArray, I::CartesianIndex)
   A.content[_offset(A, I)]
end

# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{AbstractArray{Bool}, 1}) where {DType, N}

   # We receive a single 1 dimensional array, and we must return a 1 dimension array
   mask = args[1] 

   # The number of elements in the mask should match the number of elements in the array
   numberOfElementsMatch(array, mask) || throw( DomainError("The number of elements in the mask and the array should be equal") )

   # We then filter the elements based on the mask
   filteredArray = Vector{DType}()

   for (i, val) in enumerate(array)
      mask[i] && push!(filteredArray, val)
   end

   # And return a new ndarray with the filtered content
   out = NDArray{DType}(filteredArray)
   return out
end

function findFinalShape(shape::Tuple, args::Vararg{Union{AbstractArray{Bool}, Colon}})
   finalShape = Vector{Int}()

   for (i, boolArray) in enumerate(args)
      if (isa(boolArray, Colon))
         push!(finalShape, shape[i])
      else
         numberOfTrue = length(filter(x -> x == true, boolArray))         
         if (numberOfTrue != 0)
            push!(finalShape, numberOfTrue)
         end
      end
   end
   return tuple(finalShape)
end


# ndarray[[1, 2, 3]] (fancy indexing)
function Base.getindex(array::NDArray{DType, N}, args::Vararg{Union{AbstractArray{Bool}, Colon}}) where {DType, N}

   # We receive a single 1 dimensional array, and we must return a 1 dimension array
   mask = args[1] 

   shape = size(array)

   # The number of elements in the mask should match the number of elements in the array
   for (i, val) in enumerate(args)
      (isa(val, Colon) || numberOfElementsMatch(shape[i], val)) || throw( DomainError("The number dimensions of the mask should match") )
   end

    max_ndim = maximum(ndims.(masks))
    padded_shapes = [
        ntuple(i -> i <= ndims(mask) ? size(mask, i) : 1, max_ndim)
        for mask in masks
    ]

   # Step 2: final broadcasted shape
   final_shape = findFinalShape(shape, args)
   
   # Step 5: loop
    for idx in CartesianIndices(final_shape)
        # Compute the projected indices for each mask
        projected_indices = [project_index_function(idx, padded_shapes[j]) for j in 1:length(masks)]

        # Look up the actual integer indices from the masks
        real_indices = map(j -> masks[j][projected_indices[j]], 1:length(masks))

        # Use splatting to index into your NDArray
        output[idx] = array[real_indices...]
    end


   # We then filter the elements based on the mask
   filteredArray = Vector{DType}()

   for (i, val) in enumerate(array)
      mask[i] && push!(filteredArray, val)
   end

   # And return a new ndarray with the filtered content
   out = NDArray{DType}(filteredArray)
   return out
end
