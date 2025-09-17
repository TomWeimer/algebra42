struct RowMajorIndices
    dims::NTuple{N,Int} where N
end

function Base.iterate(R::RowMajorIndices, state=ntuple(_ -> 1, length(R.dims)))
    inds = state
    N = length(R.dims)

    # stop condition
    if inds === nothing
        return nothing
    end

    # prepare next state
    next = collect(inds)
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