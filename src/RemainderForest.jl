"""
    remainder_forest(M, m, k; kbase=0, indices=nothing, V=nothing, ans=nothing, kappa=nothing, cutoff=nothing, projective=false)

Compute modular reductions of matrix products using a remainder forest.

# Arguments
- `M`: a matrix of polynomials with integer coefficients.
- `m`: a list or dict of integers, or a function.
- `k`: a list or dict of integers, or a function. This list must be strictly monotone.
- `kbase`: an integer (defaults to 0).
- `indices`: a list of arbitrary values (optional). If included, `m` and `k` are treated
  as functions to be evaluated on these indices.
- `V`: a matrix of integers (optional). If omitted, use the identity matrix.
- `ans`: a dict of matrices (optional).
- `kappa`: a tuning parameter (optional).
- `cutoff`: an integer (optional). If specified, answers are truncated to this many columns.
- `projective`: a boolean (optional). If true, the answer is allowed to be off by a scalar multiple.

# Output
If `ans` is omitted, a dict indexed by `indices` (or by default `0:length(m)-1`) in which
`l[i] == V * prod(M(j) for j in kbase:k[i]-1) mod m[i]`.
"""
function remainder_forest(M, m, k; kbase=0, indices=nothing, V=nothing, ans=nothing,
                          kappa=nothing, cutoff=nothing, projective=false)
    error("remainder_forest: not yet implemented")
end
