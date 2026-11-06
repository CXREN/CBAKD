/// Dense numerical linear algebra.
///
/// CBAKD is a flat, single-package library of double-precision dense numerical
/// linear algebra. The `Matrix` and `Vector` value types are joined directly by
/// the algorithms layered on top of them — LU (with partial pivoting),
/// Cholesky, Householder QR, linear-system solvers, least squares, a symmetric
/// eigenvalue decomposition, the singular value decomposition, and principal
/// component analysis — with no internal sub-package boundaries.

///|
/// The package version.
pub fn version() -> String {
  "0.1.0"
}
