import Pkg

@info "Loading project"

Pkg.activate("$(@__DIR__)/..")
Pkg.instantiate()
Pkg.precompile()

cd("$(@__DIR__)/..")
include("rurki.jl")

@info "Ready"
println("Enter q to quit")

while true
    print("Enter number of elements: ")
    flush(stdout)
    input = readline()
    if input[1] == 'q'
        break
    end
    N = parse(Int, input)
    @info "Running FEM"
    rurki.fem.main(N)
    @info "Done"
end

@info "Exiting"
