using RForest
using Oscar
using Test
import Primes

"""
    batch_factorial(n, e, gamma)

Amortized computation of factorials.

# Arguments
- `n`: bound on primes
- `e`: order of the series expansion
- `gamma`: a rational number

# Output
A dict whose value at a prime `p` equals `(ceil(gamma*p) - 1)! mod p^e`.

# Examples
```julia-repl
julia> batch_factorial(10, 2, 1//2)
Dict(2 => 1, 3 => 1, 5 => 2, 7 => 6)
```
"""
function batch_factorial(n, e, gamma)
    R, k = polynomial_ring(ZZ, :k)
    M = matrix(R, 1, 1, [k])
    a, b = numerator(gamma), denominator(gamma)
    k_func = p -> -(-a * p ÷ b)
    m_func = p -> p^e
    primes = Primes.primes(2, n - 1)

    ans = remainder_forest(M, m_func, k_func; kbase=1, indices=primes)
    return Dict(p => ans[p][1, 1] for p in primes)
end

@testset "batch_factorial" begin
    result10 = batch_factorial(10, 2, 1//2)
    @test result10[2] == 1
    @test result10[3] == 1
    @test result10[5] == 2
    @test result10[7] == 6

    result100 = batch_factorial(100, 2, 1//2)
    expected100 = Dict(2=>1, 3=>1, 5=>2, 7=>6, 11=>120, 13=>44, 17=>149, 19=>75,
                       23=>47, 29=>766, 31=>1, 37=>882, 41=>706, 43=>429, 47=>2208,
                       53=>500, 59=>473, 61=>2390, 67=>1071, 71=>4971, 73=>3239,
                       79=>1263, 83=>2076, 89=>6531, 97=>6618)
    @test all(result100[p] == v for (p, v) in expected100)
end
