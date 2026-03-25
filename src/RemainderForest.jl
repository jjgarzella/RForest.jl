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

# Build M1 from a polynomial matrix M.
# Returns (dim, deg, M1::Ptr{Cvoid}) — caller must free M1 via _free_mpz!(M1, dim*dim*(deg+1)).
function _build_M1_from_poly(M)
    dim = nrows(M)
    deg = 0
    for i in 1:dim, j in 1:dim
        entry = M[i, j]
        if !(entry isa Union{ZZRingElem, Integer})
            deg = max(deg, degree(entry))
        end
    end
    deg = max(deg, 0)

    M1 = _mpz_alloc(dim * dim * (deg + 1))
    t = 0
    for i in 1:dim, j in 1:dim
        for c in 0:deg
            _mpz_init_set!(_mpz_ptr(M1, t), _poly_coeff(M[i, j], c))
            t += 1
        end
    end
    return dim, deg, M1
end

# Build M1 from a vector of coefficient matrices.
# coeffs[1] = A0 (constant), coeffs[2] = A1 (linear), etc.
# get_entry(mat, i, j) -> BigInt extracts an entry from a coefficient matrix.
# Returns (dim, deg, M1::Ptr{Cvoid}) — caller must free M1 via _free_mpz!(M1, dim*dim*(deg+1)).
function _build_M1_from_coeffs(coeffs, get_entry)
    isempty(coeffs) && throw(ArgumentError("coeffs must be non-empty"))
    dim = size(coeffs[1], 1)
    deg = length(coeffs) - 1

    M1 = _mpz_alloc(dim * dim * (deg + 1))
    t = 0
    for i in 1:dim, j in 1:dim
        for c in 0:deg
            _mpz_init_set!(_mpz_ptr(M1, t), get_entry(coeffs[c + 1], i, j))
            t += 1
        end
    end
    return dim, deg, M1
end

# ---- Shared helpers for building get_mk and idx_list ----

function _vec_get_mk(m, k)
    n = length(m)
    get_mk = t -> (BigInt(m[t]), Clong(k[t]))
    return n, get_mk, 1:n
end

function _func_get_mk(m_func, k_func, indices)
    idx_vec = collect(indices)
    n = length(idx_vec)
    get_mk = t -> (BigInt(m_func(idx_vec[t])), Clong(k_func(idx_vec[t])))
    return n, get_mk, idx_vec
end

# ---- Shared dispatch helpers ----

function _rf_vec(dim, deg, M1, m::AbstractVector, k::AbstractVector;
                  kbase, V, kappa, cutoff)
    n, get_mk, idx_list = _vec_get_mk(m, k)
    return _remainder_forest_impl(dim, n, deg, M1, get_mk;
                                   kbase, V, ans=nothing, idx_list, kappa, cutoff)
end

function _rf_func(dim, deg, M1, m::Function, k::Function, indices;
                   kbase, V, kappa, cutoff)
    n, get_mk, idx_vec = _func_get_mk(m, k, indices)
    mats = _remainder_forest_impl(dim, n, deg, M1, get_mk;
                                   kbase, V, ans=nothing, idx_list=idx_vec, kappa, cutoff)
    return Dict(zip(idx_vec, mats))
end

function _rf_vec!(ans, dim, deg, M1, m::AbstractVector, k::AbstractVector, idx_list;
                   kbase, V, kappa, cutoff)
    n, get_mk, _ = _vec_get_mk(m, k)
    _remainder_forest_impl(dim, n, deg, M1, get_mk;
                            kbase, V, ans, idx_list, kappa, cutoff)
    return ans
end

function _rf_func!(ans, dim, deg, M1, m::Function, k::Function, indices;
                    kbase, V, kappa, cutoff)
    n, get_mk, idx_vec = _func_get_mk(m, k, indices)
    _remainder_forest_impl(dim, n, deg, M1, get_mk;
                            kbase, V, ans, idx_list=idx_vec, kappa, cutoff)
    return ans
end

# ---- Polynomial matrix public methods ----

"""
    remainder_forest(M, m, k; kbase=0, V=nothing, kappa=nothing, cutoff=nothing)

Compute modular reductions of matrix products using a remainder forest.
`M` is a matrix of polynomials; `m` and `k` are vectors; returns a 1-based `Vector`.

`result[i] == V * prod(M(j) for j in kbase:k[i]-1) mod m[i]`
"""
function remainder_forest(M, m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_poly(M)
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

"""
    remainder_forest(M, m, k, indices; kbase=0, V=nothing, kappa=nothing, cutoff=nothing)

Compute modular reductions of matrix products using a remainder forest.
`M` is a matrix of polynomials; `m` and `k` are functions over `indices`; returns a `Dict`.

`result[x] == V * prod(M(j) for j in kbase:k(x)-1) mod m(x)`
"""
function remainder_forest(M, m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_poly(M)
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

"""
    remainder_forest!(ans, M, m, k; kbase=0, V=nothing, kappa=nothing, cutoff=nothing)

Mutating form: updates `ans[i] *= result[i]` for `i in 1:length(m)`.
"""
function remainder_forest!(ans, M, m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_poly(M)
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

"""
    remainder_forest!(ans, M, m, k, indices; kbase=0, V=nothing, kappa=nothing, cutoff=nothing)

Mutating form: updates `ans[x] *= result[x]` for each `x` in `indices`.
"""
function remainder_forest!(ans, M, m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_poly(M)
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: degree-1, Matrix{BigInt} ----

function remainder_forest(A0::Matrix{BigInt}, A1::Matrix{BigInt},
                           m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> mat[i, j])
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(A0::Matrix{BigInt}, A1::Matrix{BigInt},
                           m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> mat[i, j])
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0::Matrix{BigInt}, A1::Matrix{BigInt},
                            m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> mat[i, j])
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0::Matrix{BigInt}, A1::Matrix{BigInt},
                            m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> mat[i, j])
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: degree-1, MatElem{ZZRingElem} ----

function remainder_forest(A0::MatElem{ZZRingElem}, A1::MatElem{ZZRingElem},
                           m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(A0::MatElem{ZZRingElem}, A1::MatElem{ZZRingElem},
                           m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0::MatElem{ZZRingElem}, A1::MatElem{ZZRingElem},
                            m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0::MatElem{ZZRingElem}, A1::MatElem{ZZRingElem},
                            m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: degree-1, generic fallback ----

function remainder_forest(A0, A1, m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(A0, A1, m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0, A1, m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, A0, A1, m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs([A0, A1], (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: vector of coeffs, Vector{Matrix{BigInt}} ----

function remainder_forest(coeffs::Vector{Matrix{BigInt}}, m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> mat[i, j])
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(coeffs::Vector{Matrix{BigInt}}, m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> mat[i, j])
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::Vector{Matrix{BigInt}}, m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> mat[i, j])
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::Vector{Matrix{BigInt}}, m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> mat[i, j])
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: vector of coeffs, Vector{<:MatElem{ZZRingElem}} ----

function remainder_forest(coeffs::Vector{<:MatElem{ZZRingElem}}, m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(coeffs::Vector{<:MatElem{ZZRingElem}}, m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::Vector{<:MatElem{ZZRingElem}}, m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::Vector{<:MatElem{ZZRingElem}}, m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Integer-matrix public methods: vector of coeffs, generic fallback ----

function remainder_forest(coeffs::AbstractVector, m::AbstractVector, k::AbstractVector;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec(dim, deg, M1, m, k; kbase, V, kappa, cutoff)
end

function remainder_forest(coeffs::AbstractVector, m::Function, k::Function, indices;
                           kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func(dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::AbstractVector, m::AbstractVector, k::AbstractVector;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_vec!(ans, dim, deg, M1, m, k, 1:length(m); kbase, V, kappa, cutoff)
end

function remainder_forest!(ans, coeffs::AbstractVector, m::Function, k::Function, indices;
                            kbase=0, V=nothing, kappa=nothing, cutoff=nothing)
    dim, deg, M1 = _build_M1_from_coeffs(coeffs, (mat, i, j) -> BigInt(mat[i, j]))
    return _rf_func!(ans, dim, deg, M1, m, k, indices; kbase, V, kappa, cutoff)
end

# ---- Core implementation ----

# Shared by all public methods. Takes pre-built (dim, deg, M1) and frees M1 in finally.
# get_mk(t) -> (BigInt modulus, Clong k-value) for 1-based position t.
# idx_list: keys for ans mutation (only used when ans !== nothing).
# Returns Vector{MatElem{ZZRingElem}} (1-based), or nothing if ans is provided.
function _remainder_forest_impl(dim, n, deg, M1, get_mk;
                                  kbase=0, V=nothing, ans=nothing, idx_list=nothing,
                                  kappa=nothing, cutoff=nothing)

    dim == 0 && throw(ArgumentError("M must be non-empty"))

    rows = if V === nothing
        dim
    else
        size(V, 2) == dim || throw(ArgumentError("Matrix dimension mismatch"))
        size(V, 1)
    end

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

    for t in 1:n
        mv, kv = get_mk(t)
        _mpz_init_set!(_mpz_ptr(m1, t - 1), mv)
        n_m1_inited += 1
        k1[t] = kv
        if (t == 1 && k1[1] < kbase) || (t > 1 && k1[t] < k1[t-1])
            errorflag = true; break
        end
    end

    if errorflag
        for i in 0:n_m1_inited-1
            _mpz_clear!(_mpz_ptr(m1, i))
        end
        Libc.free(m1)
        _free_mpz!(M1, dim * dim * (deg + 1))
        throw(ArgumentError("k must be a monotone sequence of values not less than kbase"))
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

    local result_mats
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

    finally
        _free_mpz!(M1, dim * dim * (deg + 1))
        _free_mpz!(V1, rows * dim)
        _free_mpz!(m1, n)
        _free_mpz!(A1, rows * dim * n)
        _free_mpz!(z, 1)
    end

    return ansdict ? nothing : result_mats
end
