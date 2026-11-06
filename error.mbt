/// Errors raised by the linear-algebra routines.
///
/// Operations that can fail on the *shape or content* of their inputs — rather
/// than on programmer error — raise one of these so callers can recover.
/// Additional variants (singular matrices, failed convergence, ...) are added
/// alongside the algorithms that introduce them.
pub(all) suberror LinAlgError {
  /// Matrix and/or vector dimensions do not agree.
  DimensionMismatch
}
