/// Singular value decomposition via one-sided Jacobi orthogonalization.
///
/// `svd(a)` factors a tall (`m ≥ n`) matrix as `A = U·diag(s)·Vᵀ`, where `U`
/// (`m × n`) has orthonormal columns, `s` holds the singular values in
/// descending order, and `Vᵀ` (`n × n`) is orthogonal. The one-sided Jacobi
/// method drives the columns of `A` to mutual orthogonality with Givens
/// rotations, so the singular values are the resulting column norms — a
/// scheme that is accurate even when singular values are tiny. Raises
/// `DimensionMismatch` when the matrix has more columns than rows.

pub fn svd(a : Matrix) -> (Matrix, Vector, Matrix) raise LinAlgError {
  let m = a.rows()
  let n = a.cols()
  if m < n {
    raise DimensionMismatch
  }
  let u = a.copy()
  let v = Matrix::identity(n)
  let tol = 1.0e-12
  let max_iter = n * n * 50
  let mut iter = 0
  let mut done = false
  while !done && iter < max_iter {
    done = true
    for p in 0..<n {
      for q in (p + 1)..<n {
        // Gram entries for columns p and q.
        let mut alpha = 0.0
        let mut beta = 0.0
        let mut gamma = 0.0
        for i in 0..<m {
          let ap = u.get(i, p)
          let aq = u.get(i, q)
          alpha = alpha + ap * ap
          beta = beta + aq * aq
          gamma = gamma + ap * aq
        }
        let scale = alpha.sqrt() * beta.sqrt()
        if gamma.abs() > tol * scale {
          done = false
          let zeta = (beta - alpha) / (2.0 * gamma)
          let t = if zeta >= 0.0 {
            1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
          } else {
            -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
          }
          let c = 1.0 / (1.0 + t * t).sqrt()
          let s = t * c
          // Rotate columns p and q of the working matrix.
          for i in 0..<m {
            let up = u.get(i, p)
            let uq = u.get(i, q)
            u.set(i, p, c * up - s * uq)
            u.set(i, q, s * up + c * uq)
          }
          // Accumulate the right rotations into V.
          for i in 0..<n {
            let vp = v.get(i, p)
            let vq = v.get(i, q)
            v.set(i, p, c * vp - s * vq)
            v.set(i, q, s * vp + c * vq)
          }
        }
      }
    }
    iter = iter + 1
  }
  // Singular values are the column norms; normalize the columns into U.
  let s = Vector::zeros(n)
  for j in 0..<n {
    let mut norm = 0.0
    for i in 0..<m {
      norm = norm + u.get(i, j) * u.get(i, j)
    }
    s.set(j, norm.sqrt())
  }
  let um = Matrix::zeros(m, n)
  for j in 0..<n {
    let nj = s.get(j)
    for i in 0..<m {
      um.set(i, j, if nj > 0.0 { u.get(i, j) / nj } else { 0.0 })
    }
  }
  // Sort singular values descending, permuting U and V columns to match.
  let idx = Array::make(n, 0)
  for i in 0..<n {
    idx[i] = i
  }
  for i in 0..<n {
    let mut best = i
    for k in (i + 1)..<n {
      if s.get(idx[k]) > s.get(idx[best]) {
        best = k
      }
    }
    let t = idx[i]
    idx[i] = idx[best]
    idx[best] = t
  }
  let ss = Vector::zeros(n)
  let us = Matrix::zeros(m, n)
  let vs = Matrix::zeros(n, n)
  for j in 0..<n {
    let src = idx[j]
    ss.set(j, s.get(src))
    for i in 0..<m {
      us.set(i, j, um.get(i, src))
    }
    for i in 0..<n {
      vs.set(i, j, v.get(i, src))
    }
  }
  (us, ss, vs.transpose())
}
