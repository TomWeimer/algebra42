struct NestedArrayIndices
    dims::NTuple{N,Int} where N
end

function Base.iterate(R::NestedArrayIndices, state=ntuple(_ -> 1, length(R.dims)))
    inds = state
    N = length(R.dims) # 3

    # stop condition
    if inds === nothing # (1, 1, 1)
        return nothing
    end

    # prepare next state
    next = collect(inds) # [1, 1, 1]

    if next[end-1] < R.dims[end-1]
        next[end-1] += 1
        return (CartesianIndex(inds), Tuple(next)) 
    elseif (next[end] < R.dims[end])
        next[end-1] = 1
        next[end] += 1
        return (CartesianIndex(inds), Tuple(next)) 
    end

    for d in N:-1:1   # rightmost first
        if next[d] < R.dims[d]
            next[d] += 1
            return (CartesianIndex(inds), Tuple(next))
        else
            next[d] = 1
        end
    end
    return (CartesianIndex(inds), nothing)
end