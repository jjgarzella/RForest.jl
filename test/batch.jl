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
    result = batch_factorial(10, 2, 1//2)
    @test result[2] == 1
    @test result[3] == 1
    @test result[5] == 2
    @test result[7] == 6
end
