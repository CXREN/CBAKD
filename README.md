# CBAKD

Dense numerical linear algebra in pure MoonBit — `Matrix` and `Vector` value
types, LU / Cholesky / QR factorizations, linear and least-squares solvers, a
symmetric eigensolver, the singular value decomposition, and principal
component analysis. No native code and no FFI; the whole library compiles to
WASM.

## Install

```sh
moon add CXREN/CBAKD
```

## Quick start

Solve a square system, then fit a line in the least-squares sense:

```moonbit
let a = @cbakd.Matrix::new(2, 2, [3.0, 1.0, 1.0, 2.0])
let b = @cbakd.Vector::new([5.0, 5.0])
let x = @cbakd.solve(a, b)               // x = [1, 2]

// Fit y = 2 + 3x from four noisy-free points (intercept column + x column).
let design = @cbakd.Matrix::new(4, 2, [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
let y = @cbakd.Vector::new([2.0, 5.0, 8.0, 11.0])
let fit = @cbakd.lstsq(design, y)        // fit = [2, 3]
```

## What's inside

### Core types

- **`Matrix`** — dense row-major doubles: `new`, `zeros`, `filled`,
  `identity`, `get`/`set`, `copy`, `transpose`, `column`, `swap_rows`,
  `mul`, `mul_vec`, `add`, `sub`, `scale`, and the Frobenius / 1 / ∞ norms.
- **`Vector`** — dense doubles: `new`, `zeros`, `filled`, `get`/`set`, `dot`,
  `norm`, `scale`, `add`, `sub`, `axpy`, `to_string`.
- **`LinAlgError`** — `DimensionMismatch`, `Singular`, `NotPositiveDefinite`.

### Solvers

- `solve(a, b)` — square system `A·x = b` via LU with partial pivoting.
- `lstsq(a, b)` — least-squares `min ‖A·x − b‖₂` via Householder QR.

### Factorizations

- `lu(a)` → `LU`, with `LU::solve`, `LU::det`, `LU::inverse`.
- `cholesky(a)` → lower `L` with `A = L·Lᵀ`; also `cholesky_solve(a, b)`.
- `qr(a)` → `QR`, an orthogonal `Q` and upper-triangular `R`; also `QR::solve`.

### Decompositions & analysis

- `eig_sym(a)` → `(values, vectors)` — a symmetric eigendecomposition.
- `svd(a)` → `(U, s, Vᵀ)` — `A = U·diag(s)·Vᵀ`, singular values descending.
- `pca(x)` → `PCA` — feature means, orthonormal components, per-component
  variance, plus `PCA::transform` for centering and projecting new data.

## Run the demo

```sh
moon run example
```

## Notes on the numerics

- LU uses partial pivoting; a zero pivot raises `Singular`.
- `qr` and `svd` expect a tall matrix (`m ≥ n`); `pca` needs at least two
  samples.
- The eigensolver and SVD are Jacobi methods: unconditionally convergent for
  symmetric matrices, and accurate even for very small singular values.

## License

Apache-2.0.
