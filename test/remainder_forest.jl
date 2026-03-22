using RForest
using Oscar
using Test
import Primes

@testset "remainder_forest" begin

    R, x = polynomial_ring(ZZ, :x)

    # Wilson's theorem: (p-1)! ≡ -1 (mod p) for every prime p.
    # The primes where the stronger (p-1)! ≡ -1 (mod p^2) holds are
    # called Wilson primes; the only known ones below 600 are 5, 13, 563.
    @testset "Wilson primes, list style" begin
        M = matrix(R, 1, 1, [x])
        primes600 = Primes.primes(2, 599)
        m = [p^2 for p in primes600]
        k = [p  for p in primes600]
        l = remainder_forest(M, m, k; kbase=1)
        n = length(primes600)

        @test all(mod(l[i][1,1], k[i+1]) == k[i+1] - 1 for i in 0:n-1)
        @test [k[i+1] for i in 0:n-1 if l[i][1,1] == m[i+1] - 1] == [5, 13, 563]
    end

    # Same computation using the index/function calling style.
    @testset "Wilson primes, index style" begin
        M = matrix(R, 1, 1, [x])
        primes600 = Primes.primes(2, 599)
        d = remainder_forest(M, p -> p^2, p -> p; kbase=1, indices=primes600)

        @test all(mod(d[p][1,1], p) == p - 1 for p in primes600)
        @test [p for p in primes600 if d[p][1,1] == p^2 - 1] == [5, 13, 563]
    end

    # The matrix [[1,0],[x,1]] with initial row [0,1] accumulates a running sum:
    # the (1,1) entry of result[i] equals 0+1+...+i mod (i+1), and the (1,2)
    # entry is always 1 (for i >= 1).
    @testset "running sum" begin
        M = matrix(R, 2, 2, [R(1), R(0), x, R(1)])
        m = collect(1:99)
        k = collect(1:99)
        V = matrix(ZZ, 1, 2, [0, 1])
        l = remainder_forest(M, m, k; V=V)
        n = length(m)

        @test all(l[i][1,2] == 1 for i in 1:n-1)
        @test all(l[i][1,1] == (iseven(i) ? 0 : (i+1)÷2) for i in 0:n-1)
    end

end
