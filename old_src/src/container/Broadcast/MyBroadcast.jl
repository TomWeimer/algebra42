macro MyBroadcast(expr)
    # Replace broadcast operator
    function broadcast_symbol_op(s)
        return s == :.+ || s == :.- || s == :./ || s == :.* || s == :.> || s == :.<
    end

    function is_broadcast_op(ex)
        return ( ex.head == :call && broadcast_symbol_op(ex.args[1]) )  || ex.head == :. 
    end

    function _wrap_broadcast(expr)
        if expr isa Expr && is_broadcast_op(expr)
            op = get(op_map, expr.args[1], nothing)
            if !(isnothing(op))
                return Expr(:call, :Broadcasted42, op, expr.args[2:end]...)
            end
            return Expr(:call, :Broadcasted42, BroadcastedFunction42(getfield(Main, expr.args[1])), expr.args[2:end]...)
        end
        return expr
    end



    function _replace_broadcast(expr)
        if (expr isa Expr)
            args = map(_replace_broadcast, expr.args)
            new_args = map(_wrap_broadcast, args)
            expr.args = new_args
        end
        return expr
    end

    function replace_broadcast(expr)
        expr = _replace_broadcast(expr)
        # We check if the first expr is broadcast
        if (expr isa Expr && is_broadcast_op(expr))
            return _wrap_broadcast(expr)
        end
        return expr
    end

    # Add materialize
    function is_inplace_op(s)
        return s == :(.=) || s == :(.*=) || s == :(./=) || s == :(.-=) || s == :(.+=)
    end

    function arg_is_broadcasted(arg)
        if (arg isa Expr)
            return arg.head == :call && arg.args[1] == :Broadcasted42
        end
        return false
    end

    function args_broadcasted(args...)
        mat_indices = []
        for (i, arg) in enumerate(args)
            if arg_is_broadcasted(arg)
                push!(mat_indices, i)
            end
        end
        return mat_indices
    end

    function which_need_materialize(expr)
        if (expr isa Expr)
            head = expr.head
            return args_broadcasted(expr.args...)

        end
        return []
    end

    # function __add_materialize(head, arg)
    #     if (is_inplace_op(head))
    #         op = op_map[head]
    #         return op === nothing ? Expr(:call, :materialize42!, arg) : Expr(:call, :materialize42!, op, arg)
    #     end
    #     return Expr(:call, :materialize42, arg)
    # end

    function _add_materialize(expr)
        if (expr isa Expr)
            indices_needing_materialize = which_need_materialize(expr)
            if !isempty(indices_needing_materialize)
                if (is_inplace_op(expr.head))
                    op = op_map[expr.head]
                    return op === nothing ? Expr(:call, :materialize42!, expr.args...) : Expr(:call, :materialize42!, op, expr.args...)
                end
                for i in indices_needing_materialize
                    expr.args[i] = Expr(:call, :materialize42, expr.args[i])
                end
                return expr
            end
            expr.args = map(_add_materialize, expr.args)
        end
        return expr
    end

    function add_materialize(expr)
        # Check the firs expression directly:
        if (expr isa Expr && expr.head == :call && expr.args[1] == :Broadcasted42)
                if (is_inplace_op(expr.head))
                    op = op_map[expr.head]
                    return op === nothing ? Expr(:call, :materialize42!, expr) : Expr(:call, :materialize42!, op, expr)
                end
                return Expr(:call, :materialize42, expr)
        end
        return _add_materialize(expr)
    end

    rtn = replace_broadcast(expr)

    # println("adding broadcasted: ", rtn)

    rtn = add_materialize(rtn)

    #
    println("adding materialize: ", rtn)
    return esc(rtn)
end
