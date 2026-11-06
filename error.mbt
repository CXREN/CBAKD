///|
/// Errors raised by the linear-algebra routines.
///
/// Operations that can fail on the *shape or content* of their inputs — rather
/// than on programmer error — raise one of these so callers can recover.
/// Additional variants (singular matrices, failed convergence, ...) are added
/// alongside the algorithms that introduce them.
pub(all) suberror LinAlgError {
  /// Matrix and/or vector dimensions do not agree.
  DimensionMismatch
  /// A matrix is singular or rank-deficient where an inverse or unique
  /// solution is required (for example a zero pivot in LU).
  Singular
  /// Cholesky factorization was asked of a matrix that is not symmetric
  /// positive-definite (a non-positive diagonal pivot).
  NotPositiveDefinite
}
