```@meta
CurrentModule = RForest
```

# RForest.jl

RForest.jl is a Julia wrapper around Sutherland's [`rforest`](https://github.com/edgarcosta/rforest)
library for computing **remainder forests** — a technique for evaluating many modular
matrix products simultaneously in average polynomial time.

The algorithm is described in [Costa, Gerbicz, and Harvey (2012)](https://arxiv.org/abs/1209.3436)
(primary reference), with further development in
[Harvey (2012)](https://arxiv.org/abs/1210.8239),
[Harvey (2014)](https://arxiv.org/abs/1402.3439), and
[Harvey and Sutherland (2014)](https://arxiv.org/abs/1410.5222).

!!! note "AI disclosure"
    This documentation and the Julia wrapper code were written with the assistance
    of Claude (Anthropic).

## What is a remainder forest?

Given a matrix polynomial `M(x)` and a list of pairs `(mᵢ, kᵢ)`, a remainder forest
computes

```
result[i] = V * M(kbase) * M(kbase+1) * ⋯ * M(kᵢ-1)  mod mᵢ
```

for every `i` at once, where `V` is an optional starting matrix (identity by default).
The naive approach computes each product independently; a remainder forest exploits the
shared prefix structure to do the work in `O(n log n)` matrix multiplications instead of
`O(n · max(kᵢ))`.

## Quick example: Wilson primes

Wilson's theorem says `(p-1)! ≡ -1 (mod p)` for every prime `p`. The stronger condition
`(p-1)! ≡ -1 (mod p²)` defines the **Wilson primes**; the only known ones below 600 are
5, 13, and 563.

We can find them by computing `(p-1)! mod p²` for all primes `p < 600` at once.
The factorial is a matrix product of the 1×1 polynomial matrix `[x]` evaluated
at `x = 1, 2, …, p-1`:

```julia
using RForest, Oscar
import Primes

R, x = polynomial_ring(ZZ, :x)
M = matrix(R, 1, 1, [x])          # 1×1 polynomial matrix: M(j) = j

primes600 = Primes.primes(2, 599)
m = [p^2 for p in primes600]      # moduli
k = [p   for p in primes600]      # upper limits (exclusive)

# result[i][1,1] == (p-1)! mod p²  where p = primes600[i]
result = remainder_forest(M, m, k; kbase=1)

wilson_primes = [primes600[i] for i in eachindex(primes600)
                 if result[i][1,1] == m[i] - 1]
# [5, 13, 563]
```

The same computation using the function/index calling style, which returns a `Dict` keyed
by the primes directly:

```julia
d = remainder_forest(M, p -> p^2, p -> p, primes600; kbase=1)

wilson_primes = [p for p in primes600 if d[p][1,1] == p^2 - 1]
# [5, 13, 563]
```

## Passing integer matrices directly

If your data is already in matrix form you can skip the polynomial representation and pass
the coefficient matrices `A₀, A₁` of `M(x) = A₀ + A₁x` directly:

```julia
# M(x) = [[x]] has A₀ = [[0]], A₁ = [[1]]
A0 = BigInt[0;;]
A1 = BigInt[1;;]

result = remainder_forest(A0, A1, m, k; kbase=1)
```

For higher-degree polynomials pass a vector of coefficient matrices `[A₀, A₁, A₂, …]`:

```julia
result = remainder_forest([A0, A1, A2], m, k; kbase=1)
```

## API reference

```@index
```

```@autodocs
Modules = [RForest]
```
