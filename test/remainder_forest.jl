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

        @test all(mod(l[i][1,1], k[i]) == k[i] - 1 for i in 1:n)
        @test [k[i] for i in 1:n if l[i][1,1] == m[i] - 1] == [5, 13, 563]
    end

    # Same computation using the index/function calling style.
    @testset "Wilson primes, index style" begin
        M = matrix(R, 1, 1, [x])
        primes600 = Primes.primes(2, 599)
        d = remainder_forest(M, p -> p^2, p -> p, primes600; kbase=1)

        @test all(mod(d[p][1,1], p) == p - 1 for p in primes600)
        @test [p for p in primes600 if d[p][1,1] == p^2 - 1] == [5, 13, 563]
    end

    # The matrix [[1,0],[x,1]] with initial row [0,1] accumulates a running sum:
    # result[i][1,1] == sum(0:i-1) mod i == (isodd(i) ? 0 : i÷2),
    # and result[i][1,2] == 1 for i >= 2.
    @testset "running sum" begin
        M = matrix(R, 2, 2, [R(1), R(0), x, R(1)])
        m = collect(1:99)
        k = collect(1:99)
        V = matrix(ZZ, 1, 2, [0, 1])
        l = remainder_forest(M, m, k; V=V)
        n = length(m)

        @test all(l[i][1,2] == 1 for i in 2:n)
        @test all(l[i][1,1] == (isodd(i) ? 0 : i÷2) for i in 1:n)
    end

    # Integer-matrix input: M(x) = A0 + A1*x where A0=[[0]], A1=[[1]].
    # This is the same as the polynomial matrix [[x]] with kbase=1,
    # so results must match the Wilson primes list-style test.
    #
    # Note: avoid zeros(BigInt,...)/ones(BigInt,...) here. Julia's fill!-based
    # constructors store the *same* BigInt object in every slot, causing all
    # entries to alias. _build_M1_from_coeffs only reads entries (via
    # __gmpz_init_set, which copies the value), so aliasing is safe — but using
    # explicit literals makes intent clear and avoids the surprise entirely.
    primes600 = Primes.primes(2, 599)
    m600 = [p^2 for p in primes600]
    k600 = [p   for p in primes600]
    n600 = length(primes600)

    @testset "integer-matrix input, Matrix{BigInt}" begin
        A0 = BigInt[0;;]
        A1 = BigInt[1;;]
        l = remainder_forest(A0, A1, m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

    @testset "integer-matrix input, MatElem{ZZRingElem}" begin
        A0 = matrix(ZZ, 1, 1, [0])
        A1 = matrix(ZZ, 1, 1, [1])
        l = remainder_forest(A0, A1, m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

    @testset "integer-matrix input, generic fallback (Matrix{Int})" begin
        A0 = Int[0;;]
        A1 = Int[1;;]
        l = remainder_forest(A0, A1, m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

    @testset "vector-of-coefficients input, Vector{Matrix{BigInt}}" begin
        A0 = BigInt[0;;]
        A1 = BigInt[1;;]
        l = remainder_forest([A0, A1], m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

    @testset "vector-of-coefficients input, Vector{MatElem{ZZRingElem}}" begin
        A0 = matrix(ZZ, 1, 1, [0])
        A1 = matrix(ZZ, 1, 1, [1])
        l = remainder_forest([A0, A1], m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

    @testset "vector-of-coefficients input, generic fallback" begin
        A0 = Int[0;;]
        A1 = Int[1;;]
        l = remainder_forest([A0, A1], m600, k600; kbase=1)
        @test all(mod(l[i][1,1], k600[i]) == k600[i] - 1 for i in 1:n600)
        @test [k600[i] for i in 1:n600 if l[i][1,1] == m600[i] - 1] == [5, 13, 563]
    end

end
