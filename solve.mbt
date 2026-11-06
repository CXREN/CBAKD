/// Convenience solvers built on the factorizations in `factor`.
///
/// These are the entry points most callers want: solve a square linear
/// system, or fit an overdetermined system in the least-squares sense.

/// Solve the square system `A·x = b` using LU with partial pivoting.
pub fn solve(a : Matrix, b : Vector) -> Vector raise LinAlgError {
  lu(a).solve(b)
}

/// Solve the least-squares problem `min ‖A·x − b‖₂` using Householder QR.
pub fn lstsq(a : Matrix, b : Vector) -> Vector raise LinAlgError {
  qr(a).solve(b)
}
