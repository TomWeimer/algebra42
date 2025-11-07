import Base: *, +, -, /, ==

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                            BasicMath                                             #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# Basic math operations:
add(elem1::Number, elem2::Number)  = (println(stderr, "called mother fucker + "); elem1 + elem2)
sub(elem1::Number, elem2::Number)  = (println(stderr, "called mother fucker -"); elem1 - elem2)
prod(elem1::Number, elem2::Number) = (println(stderr, "called mother fucker *"); elem1 * elem2)

# Absolute value:

abs(nb::Number) = nb < 0 ? -nb : nb

# Useable with generator:

function prod(iterable)
   # println(stderr, "called mother fucker * prod ");
    result = one(eltype(iterable))  # start with multiplicative identity
    for x in iterable
        result *= x
    end
    return result
end

function sum(iter)
    println(stderr, "called mother fucker + sum ");
    state = iterate(iter)
    state === nothing && return 0  # empty iterable

    x, st = state
    s = zero(typeof(x))            # use type of first element
    s += x

    while true
        next_state = iterate(iter, st)
        next_state === nothing && break
        x, st = next_state
        s += x
    end

    return s
end

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#                                                                                                  #
#                                           ComplexType                                            #
#                                                                                                  #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

struct Complex42 <: Number
    real::Float32
    imag::Float32
end


# ──── constructor ─────────────────────────────────────────────────────────────────────────────── #

Complex42(real::Real, imag::Real) = Complex42(Float32(real), Float32(imag))
Complex42(real::Real) = Complex42(Float32(real), zero(Float32))

# ──── complex operation ───────────────────────────────────────────────────────────────────────── #

Base.real(z::Complex42) = z.real
Base.imag(z::Complex42) = z.imag
Base.conj(z::Complex42) = Complex42(z.real, -z.imag)

Base.abs2(z::Complex42) = z.real*z.real + z.imag*z.imag
Base.abs(z::Complex42) = sqrt(abs2(z))

# ──── complex arithmetic ──────────────────────────────────────────────────────────────────────── #

-(z::Complex42) = Complex42(-z.real, -z.imag)
==(a::Complex42, b::Complex42) = a.real == b.real && a.imag == b.imag

+(a::Complex42, b::Complex42) = Complex42(a.real + b.real, a.imag + b.imag)
-(a::Complex42, b::Complex42) = Complex42(a.real - b.real, a.imag - b.imag)

*(a::Complex42, b::Complex42) = Complex42(
    a.real * b.real - a.imag * b.imag, 
    a.real * b.imag + a.imag * b.real
)

/(a::Complex42, b::Complex42) = (
    # a + bi / c + di = ac+bd/(c^2 + d^2) + i(bc-ad)/(c^2 + d^2)
    denominator = abs2(b);
    Complex42(
        (a.real * b.real + a.imag * b.imag) / denominator,
        (a.imag * b.real - a.real * b.imag) / denominator
    )
)

# ──── conversions ─────────────────────────────────────────────────────────────────────────────── #

Base.convert(::Type{Complex42}, x::Real) = Complex42(x, 0)
Base.convert(::Type{Complex42}, z::Complex) = Complex42(real(z), imag(z))

Base.promote_rule(::Type{Complex42}, ::Type{<:Real}) = Complex42
Base.promote_rule(::Type{Complex42}, ::Type{<:Complex}) = Complex42

# ──── print function ──────────────────────────────────────────────────────────────────────────── #

Base.show(io::IO, z::Complex42) = print(io, "Complex42(", z.real, ", ", z.imag, ")")