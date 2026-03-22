using RForest
using Test

@testset "RForest.jl" begin
    include("batch.jl")
    include("remainder_forest.jl")
end
