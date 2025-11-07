# macro to mimic interface
macro mustoverload()
    return :(throw(NotImplementedError("This function must be overloaded by concrete subtypes")))
end

# used in index error
macro throw_index_error(indices, msg)
    return :(throw(DomainError($(esc(indices)) , $(esc(msg)))))
end

# syntaxic sugar
macro throw_error(exception, msg, args...)
    return :(throw($(esc(exception))($(esc(msg)), $(esc.(args)...))))
end