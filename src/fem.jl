module fem

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
@int 0 pi (@int 0 2sin(phi) r dr) dphi
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

function main(N::Int)
    @assert N > 1

    println("Problem definition:")
    println("-u'' - u = sin(x)")
    println("u(0) = 0")
    println("u'(2) - u(2) = 0")
    println("0 <= x <= 2")

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

    # main row
    maindiagelem  = B(_e(2), _de(2), _e(2), _de(2), xi[1], xi[3])
    upperdiagelem = B(_e(3), _de(3), _e(2), _de(2), xi[2], xi[3])

    maindiag  = fill(maindiagelem, n-1)
    upperdiag = fill(upperdiagelem, n-2)

    # last row
    maindiag[end]  = B(_e(n  ), _de(n  ), _e(n), _de(n), xi[n-1], xi[n])

    B_ij = SymTridiagonal(maindiag, upperdiag)

    L_j = Vector{Float64}(undef, n-1)
    Threads.@threads for j=2:n-1
        L_j[j-1] = L(_e(j), xi[j-1], xi[j+1])
    end
    L_j[end] = L(_e(n), xi[n-1], xi[n])

    if N < 50
        println("B | L matrix:")
        show(stdout, "text/plain", [B_ij L_j])
        println()
    end

    @info "Solving"
    u_i = B_ij \ L_j

    if N < 50
        println("u vector:")
        show(stdout, "text/plain", u_i)
        println()
    end

    # define computed function
    u(x) = sum([u_i[i-1] * e(xi, i, x) for i=2:n])

    # actual solution
    real_u(x) = 1/2 * (x * cos(x) + (sin(x) * (2sin(2) + cos(2))) / (cos(2) - sin(2)))

    # dense nodes for plotting
    xs = range(a, b, length=100)

    @info "Plotting"
    plot(xs, [u, real_u],
        labels=["Computed" "Actual"],
        title="N = $N",
    ) |> display

    print("Save plot [y/N]? ")
    flush(stdout)
    resp = readline()
    if length(resp) > 0 && resp[1] == 'y'
        print("Enter filename: ")
        flush(stdout)
        name = readline()
        @info "Saving plot to $name"
        savefig(name)
        @info "Saved"
    end

    @info "Verification"
    # mean squared error
    mse = sum( @. (u(xs) - real_u(xs))^2 ) / length(xs)
    @show mse

    nothing
end

end # module
