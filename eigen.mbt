/// Symmetric eigenvalue decomposition via cyclic Jacobi rotations.
///
/// `eig_sym(a)` returns the eigenvalues (as a `Vector`, sorted by the
/// algorithm's convergence order rather than descending) and the matrix whose
/// columns are the corresponding orthonormal eigenvectors, such that
/// `A·V = V·diag(values)`. Jacobi rotations are unconditionally convergent for
/// real symmetric matrices and are numerically very accurate for small-to-
/// medium sizes.

pub fn eig_sym(a : Matrix) -> (Vector, Matrix) raise LinAlgError {
  if !a.is_square() {
    raise DimensionMismatch
  }
  let n = a.rows()
  let m = a.copy()
  let v = Matrix::identity(n)
  let tol = 1.0e-12
  let max_iter = n * n * 50
  let mut iter = 0
  let mut done = false
  while !done && iter < max_iter {
    done = true
    for p in 0..<n {
      for q in (p + 1)..<n {
        let apq = m.get(p, q)
        if apq.abs() > tol {
          done = false
          let app = m.get(p, p)
          let aqq = m.get(q, q)
          let tau = (aqq - app) / (2.0 * apq)
          let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
          } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
          }
          let c = 1.0 / (1.0 + t * t).sqrt()
          let s = t * c
          // Rotate the off-diagonal rows/columns (k ≠ p, q).
          for k in 0..<n {
            if k != p && k != q {
              let akp = m.get(k, p)
              let akq = m.get(k, q)
              let nkp = c * akp - s * akq
              let nkq = s * akp + c * akq
              m.set(k, p, nkp)
              m.set(p, k, nkp)
              m.set(k, q, nkq)
              m.set(q, k, nkq)
            }
          }
          // Update the (p, q) diagonal block.
          m.set(p, p, c * c * app - 2.0 * s * c * apq + s * s * aqq)
          m.set(q, q, s * s * app + 2.0 * s * c * apq + c * c * aqq)
          m.set(p, q, 0.0)
          m.set(q, p, 0.0)
          // Accumulate eigenvectors: V = V · J.
          for k in 0..<n {
            let vkp = v.get(k, p)
            let vkq = v.get(k, q)
            v.set(k, p, c * vkp - s * vkq)
            v.set(k, q, s * vkp + c * vkq)
          }
        }
      }
    }
    iter = iter + 1
  }
  let w = Vector::zeros(n)
  for i in 0..<n {
    w.set(i, m.get(i, i))
  }
  (w, v)
}
