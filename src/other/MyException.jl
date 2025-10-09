module MyException

const NOT_FOUND_MSG = "The value searched was not found";

struct NotFoundError <: Exception
    msg::String
    NotFoundError() = new(NOT_FOUND_MSG)
end
    module Preconditions

        function check(mustBeTrue::Bool)
            mustBeTrue || throw(Base.ArgumentError("The pre-condition must be true"));
        end


        function check(mustBeTrue::Bool, str::String)
            mustBeTrue || throw(Base.ArgumentError(str));
        end
        
    end

    module Runtime
        function check(mustBeTrue::Bool)
            mustBeTrue || throw(Base.DomainError("The value obtained is invalid and doesn't belong to the output domain"));
        end


        function check(mustBeTrue::Bool, str::String)
            mustBeTrue || throw(Base.ArgumentError(str));
        end

        function wasFound(result)
            result ≠ nothing || throw( NotFoundError() );
        end
    end
end