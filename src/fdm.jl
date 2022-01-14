module fdm

using Plots
using LinearAlgebra

function main(N::Int)
    a = 0
    b = 2
    h = (b-a) / N
    n = N + 1

    x = range(a, b, length=n)

    B = Tridiagonal(
        Vector{Float64}(undef, n-1),
        Vector{Float64}(undef, n  ),
        Vector{Float64}(undef, n-1),
    )

    L = Vector{Float64}(undef, n)

    Threads.@threads for i = 2:n-1
        B[i, i-1] = -1/h^2
        B[i, i  ] = 2/h^2 - 1
        B[i, i+1] = -1/h^2
        L[i] = sin(x[i])
    end

    B[1, 1] = 1
    B[1, 2] = 0
    L[1] = 0

    B[n, n-1] = -2/h^2
    B[n, n  ] = 2/h^2 - 1 - 2/h
    L[n] = sin(x[n])

    u = B \ L

    real_u(x) = 1/2 * (x * cos(x) + (sin(x) * (2sin(2) + cos(2))) / (cos(2) - sin(2)))

    plot(x, u)
    plot!(range(a, b, length=100), real_u) |> display

    nothing
end

end
