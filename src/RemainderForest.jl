using rforest_jll: librforest

# Size of GMP's mpz_t struct, which matches Julia's BigInt layout
const _MPZ_SIZE = sizeof(BigInt)

# Pointer arithmetic for arrays of mpz_t stored in malloc'd C memory
_mpz_ptr(base::Ptr{Cvoid}, i::Integer) = base + i * _MPZ_SIZE

_mpz_alloc(n::Integer) = Ptr{Cvoid}(Libc.malloc(n * _MPZ_SIZE))

_mpz_init!(p::Ptr{Cvoid}) =
    ccall((:__gmpz_init, :libgmp), Cvoid, (Ptr{Cvoid},), p)

function _mpz_init_set!(dst::Ptr{Cvoid}, src::BigInt)
    GC.@preserve src ccall((:__gmpz_init_set, :libgmp), Cvoid,
                            (Ptr{Cvoid}, Ref{BigInt}), dst, src)
end

function _mpz_get(src::Ptr{Cvoid})
    r = BigInt()
    GC.@preserve r ccall((:__gmpz_set, :libgmp), Cvoid,
                          (Ref{BigInt}, Ptr{Cvoid}), r, src)
    return r
end

_mpz_clear!(p::Ptr{Cvoid}) =
    ccall((:__gmpz_clear, :libgmp), Cvoid, (Ptr{Cvoid},), p)

function _free_mpz!(ptr::Ptr{Cvoid}, n::Integer)
    for i in 0:n-1
        _mpz_clear!(_mpz_ptr(ptr, i))
    end
    Libc.free(ptr)
end

# Get the c-th polynomial coefficient of an entry as BigInt.
# Handles both polynomial ring elements and plain integers (degree-0).
function _poly_coeff(entry, c::Int)::BigInt
    if entry isa Union{ZZRingElem, Integer}
        return c == 0 ? BigInt(entry) : BigInt(0)
    else
        return BigInt(coeff(entry, c))
    end
end

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

    nrows(M) == ncols(M) || throw(ArgumentError("M must be square"))
    dim = nrows(M)

    rows = if V === nothing
        dim
    else
        ncols(V) == dim || throw(ArgumentError("Matrix dimension mismatch"))
        nrows(V)
    end

    n = indices === nothing ? length(m) : length(indices)

    # Determine maximum polynomial degree across all entries of M
    deg = 0
    for i in 1:dim, j in 1:dim
        entry = M[i, j]
        if !(entry isa Union{ZZRingElem, Integer})
            deg = max(deg, degree(entry))
        end
    end
    deg = max(deg, 0)

    kappa1 = if kappa === nothing
        n <= 1 ? 1 : ceil(Int, log2(log2(float(n)))) + 1
    else
        Int(kappa)
    end
    numcols = cutoff === nothing ? dim : Int(cutoff)

    # --- Allocate and fill m1 (moduli) and k1 (k-values) ---
    m1 = _mpz_alloc(n)
    k1 = Vector{Clong}(undef, n)
    n_m1_inited = 0
    errorflag = false

    if indices === nothing
        for t in 1:n
            _mpz_init_set!(_mpz_ptr(m1, t - 1), BigInt(m[t]))
            n_m1_inited += 1
            k1[t] = Clong(k[t])
            if (t == 1 && k1[1] < kbase) || (t > 1 && k1[t] < k1[t-1])
                errorflag = true; break
            end
        end
    else
        for (t, x) in enumerate(indices)
            mv = m isa Dict ? m[x] : m(x)
            kv = k isa Dict ? k[x] : k(x)
            _mpz_init_set!(_mpz_ptr(m1, t - 1), BigInt(mv))
            n_m1_inited += 1
            k1[t] = Clong(kv)
            if (t == 1 && k1[1] < kbase) || (t > 1 && k1[t] < k1[t-1])
                errorflag = true; break
            end
        end
    end

    if errorflag
        for i in 0:n_m1_inited-1
            _mpz_clear!(_mpz_ptr(m1, i))
        end
        Libc.free(m1)
        throw(ArgumentError("k must be a monotone sequence of values not less than kbase"))
    end

    # --- Allocate and fill M1 (polynomial coefficient array) ---
    # Layout: for each (i,j), store deg+1 coefficients consecutively.
    M1 = _mpz_alloc(dim * dim * (deg + 1))
    t = 0
    for i in 1:dim, j in 1:dim
        for c in 0:deg
            _mpz_init_set!(_mpz_ptr(M1, t), _poly_coeff(M[i, j], c))
            t += 1
        end
    end

    # --- Allocate and fill V1 (initial row matrix) ---
    V1 = _mpz_alloc(rows * dim)
    t = 0
    for i in 1:rows, j in 1:dim
        val = if V === nothing
            BigInt(i == j ? 1 : 0)
        else
            BigInt(V[i, j])
        end
        _mpz_init_set!(_mpz_ptr(V1, t), val)
        t += 1
    end

    # --- Allocate output A1 and product z ---
    A1 = _mpz_alloc(rows * dim * n)
    for t in 0:rows*dim*n-1
        _mpz_init!(_mpz_ptr(A1, t))
    end
    z = _mpz_alloc(1)
    _mpz_init!(_mpz_ptr(z, 0))

    local result_dict
    ansdict = ans !== nothing

    try
        ccall((:mproduct, librforest), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Clong),
              _mpz_ptr(z, 0), m1, Clong(n))

        ccall((:rforest, librforest), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, Cint,
               Ptr{Cvoid}, Clong, Ptr{Clong}, Clong, Ptr{Cvoid}, Cint),
              A1, V1, Cint(rows), M1, Cint(deg), Cint(dim),
              m1, Clong(kbase), k1, Clong(n), _mpz_ptr(z, 0), Cint(kappa1))

        # Retrieve results
        idx_list = collect(indices !== nothing ? indices : 0:n-1)
        result_mats = ansdict ? nothing : Vector{MatElem{ZZRingElem}}(undef, n)

        t = 0
        for i in 1:n
            entries = Vector{ZZRingElem}(undef, rows * numcols)
            for j in 1:rows
                for c in 1:numcols
                    entries[(j-1)*numcols + c] = ZZ(_mpz_get(_mpz_ptr(A1, t)))
                    t += 1
                end
                t += dim - numcols  # skip columns beyond cutoff
            end
            mat = matrix(ZZ, rows, numcols, entries)

            if ansdict
                ans[idx_list[i]] *= mat
            else
                result_mats[i] = mat
            end
        end

        if !ansdict
            result_dict = Dict(zip(idx_list, result_mats))
        end

    finally
        _free_mpz!(M1, dim * dim * (deg + 1))
        _free_mpz!(V1, rows * dim)
        _free_mpz!(m1, n)
        _free_mpz!(A1, rows * dim * n)
        _free_mpz!(z, 1)
    end

    return ansdict ? nothing : result_dict
end
