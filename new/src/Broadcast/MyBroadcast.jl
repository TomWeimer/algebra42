# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                           MyBroadcast                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# Macro redirecting broadcast operator to our broadcast system
macro MyBroadcast(expr)

# ──── replace broadcasted operators ───────────────────────────────────────────────────────────── #

    broadcast_symbol_op(s) = (
        s == :.+ || s == :.- || s == :./ || s == :.* || s == :.> || s == :.<
    )

    is_broadcast_op(ex) = (
        ( ex.head == :call && broadcast_symbol_op(ex.args[1]) )  || ex.head == :. 
    )

    function _wrap_broadcast(expr)
        if expr isa Expr && is_broadcast_op(expr)
            op = get(op_map, expr.args[1], nothing)
            if !(isnothing(op))
                return Expr(:call, :Broadcasted42, op, expr.args[2:end]...)
            end
            return Expr(
                :call, 
                :Broadcasted42, 
                BroadcastedFunction42(getfield(Main, expr.args[1])), 
                expr.args[2:end]...
            )
        end
        return expr
    end

    # We check if the other expr are broadcasted
    function _replace_broadcast(expr)
        if (expr isa Expr)
            args = map(_replace_broadcast, expr.args)
            new_args = map(_wrap_broadcast, args)
            expr.args = new_args
        end
        return expr
    end

    # We check if the first expr is broadcasted
    function replace_broadcast(expr)
        expr = _replace_broadcast(expr)   
        if (expr isa Expr && is_broadcast_op(expr))
            return _wrap_broadcast(expr)
        end
        return expr
    end

# ──── add materialize function ────────────────────────────────────────────────────────────────── #

    is_inplace_op(s) = (
        s == :(.=) || s == :(.*=) || s == :(./=) || s == :(.-=) || s == :(.+=)
    )

    arg_is_broadcasted(arg) = (
        (arg isa Expr) && arg.head == :call && arg.args[1] == :Broadcasted42
    )

    function args_broadcasted(expr)
        mat_indices = []
        if (expr isa Expr)
            for (i, arg) in enumerate(expr.args)
                if arg_is_broadcasted(arg)
                    push!(mat_indices, i)
                end
            end
        end
        return mat_indices
    end

    # We check if the other expr need a materialize
    function _add_materialize(expr)
        expr isa Expr || return expr
        
        idxs = args_broadcasted(expr)
        if !isempty(idxs)

            # Wrap indexed in place broadcast args in `materialize42!`
            if (is_inplace_op(expr.head))
                op = op_map[expr.head]
                return op === nothing ? 
                    Expr(:call, :materialize42!, expr.args...) : 
                    Expr(:call, :materialize42!, op, expr.args...)
            end

            # Wrap indexed broadcast args in `materialize42`
            for i in idxs
                expr.args[i] = Expr(:call, :materialize42, expr.args[i])
            end
            return expr
        end
        expr.args = map(_add_materialize, expr.args)
        return expr
    end
    
    # Check the first expression directly:
    function add_materialize(expr)
        if expr isa Expr && expr.head == :call && expr.args[1] == :Broadcasted42
            if (is_inplace_op(expr.head))
                op = op_map[expr.head]
                return op === nothing ? 
                    Expr(:call, :materialize42!, expr) : 
                    Expr(:call, :materialize42!, op, expr)
            end

            return Expr(:call, :materialize42, expr)
        end
        
        return _add_materialize(expr)
    end

    rtn = replace_broadcast(expr)
    rtn = add_materialize(rtn)
    return esc(rtn)
end
