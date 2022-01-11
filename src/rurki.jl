module rurki

using Plots
using LinearAlgebra

"""
Gauss-Legendre nodes
"""
const X_I = [
    0,
    sqrt(5 - 2sqrt(10/7))/3,
    -sqrt(5 - 2sqrt(10/7))/3,
    sqrt(5 + 2sqrt(10/7))/3,
    -sqrt(5 + 2sqrt(10/7))/3,
]

"""
Gauss-Legendre weights
"""
const W_I = [
    128/225,
    (322 + 13sqrt(70))/900,
    (322 + 13sqrt(70))/900,
    (322 - 13sqrt(70))/900,
    (322 - 13sqrt(70))/900,
]

"""
Integrate function f from a to b
"""
int(f, a, b) = (b-a)/2 * sum( @. W_I * f( (b-a)/2 * X_I + (a+b)/2 ) )

"""
Helper macro that lets you write expressions like
@int 0 1 x^2 dx
or
@int 0 2pi (@int 0 1 r dr) dphi
"""
macro int(a, b, expr, dvar)
    var = string(dvar)[2:end] |> Symbol
    esc(:( $int($var -> $expr, $a, $b) ))
end

"""
Base function
"""
function e(xi, i, x)
    @assert 1 <= i <= length(xi)
    @assert xi[begin] <= x <= xi[end]

    if i > 1 && xi[i-1] <= x <= xi[i]
        (x - xi[i-1]) / (xi[i] - xi[i-1])
    elseif i < length(xi) && xi[i] <= x <= xi[i+1]
        (xi[i+1] - x) / (xi[i+1] - xi[i])
    else
        0
    end
end

"""
Base function derivative
"""
function de(xi, i, x)
    @assert 1 <= i <= length(xi)
    @assert xi[begin] <= x <= xi[end]

    if i > 1 && xi[i-1] <= x <= xi[i]
        1 / (xi[i] - xi[i-1])
    elseif i < length(xi) && xi[i] <= x <= xi[i+1]
        -1 / (xi[i+1] - xi[i])
    else
        0
    end
end

# Weak form
B(u, du, v, dv, a, b) = -u(2)*v(2) + @int a b du(x)*dv(x) - u(x)*v(x) dx
L(v, a, b) = @int a b v(x)*sin(x) dx

function main(N)
    # domain
    a = 0
    b = 2

    # number of nodes = number of elements + 1
    n = N + 1

    # nodes
    xi = range(a, b, length=n)

    # base functions
    _e(i) = x -> e(xi, i, x)
    _de(i) = x -> de(xi, i, x)

    @info "Building B and L matrix"
    B_ij = SymTridiagonal(
        # main diagonal
        [B(_e(j  ), _de(j  ), _e(j), _de(j), xi[max(1, j-1)], xi[min(j+1, n)]) for j=2:n  ],
        # upper/lower diagonal
        [B(_e(j+1), _de(j+1), _e(j), _de(j), xi[max(1, j  )], xi[min(j+1, n)]) for j=2:n-1],
    )
    L_j = [L(_e(j), xi[max(1, j-1)], xi[min(j+1, n)]) for j=2:n]

    display([B_ij L_j])

    @info "Solving"
    u_i = B_ij \ L_j

    # add left boundary condition
    u_i = [0; u_i]

    @info "u"
    display(u_i)

    # define computed function
    u(x) = sum([u_i[i] * e(xi, i, x) for i=1:n])

    # actual solution
    real_u(x) = 1/2 * (x * cos(x) + (sin(x) * (2sin(2) + cos(2))) / (cos(2) - sin(2)))

    # dense nodes for plotting
    xs = range(a, b, length=10n)

    @info "Plotting"
    plot(xs, [u, real_u], legend=false) |> display

    @info "Verification"
    # mean squared error
    mse = sum( @. (u(xs) - real_u(xs))^2 ) / length(xs)
    @show mse

    nothing
end

end # module

# rurki.main.(3)
