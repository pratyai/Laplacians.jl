using SparseArrays
using Arpack

function construct_random_adjmatrix(n::Int)
    local adj = rand(n, n)
    adj = adj - Diagonal(diag(adj))
    adj = sparse(adj + adj')
    return adj
end

@testset "ApproxCholFactorisations" begin
    Random.seed!(1)
    local n = 10
    local adj = construct_random_adjmatrix(n)
    local split = 100000  # == merge

    # Test that the factorization is approximately correct for various
    # strategies and configurations. I.e. `lfac * lfac^T` approaches
    # `lap(adj)` in all but one element (the last remaining vertex).
    # This is expected to hold only for a large number of splits though.

    local llmat = Laplacians.LLmatp(adj, split)
    @time local ldli = Laplacians.approxChol(llmat, split, split)
    local lfac = Laplacians.ldli2Chol(ldli)'
    local llerr = lfac * lfac' - lap(adj)
    llerr[argmax(abs.(llerr))] = 0  # fix that one element
    local lambda_max, _ = eigs(llerr)
    local lambda_max = abs(first(lambda_max))
    @test lambda_max < 1e-1  # should go to 0 with larger splits

    local llmat = Laplacians.LLMatOrd(adj, split)
    @time local ldli = Laplacians.approxChol(llmat, split)
    local lfac = Laplacians.ldli2Chol(ldli)'
    local llerr = lfac * lfac' - lap(adj)
    llerr[argmax(abs.(llerr))] = 0  # fix that one element
    local lambda_max, _ = eigs(llerr)
    local lambda_max = abs(first(lambda_max))
    @test lambda_max < 1e-1  # should go to 0 with larger splits
end
