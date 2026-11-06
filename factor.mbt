///|
/// LU factorization with partial pivoting.
///
/// `lu(a)` factors a square matrix `A` as `P·A = L·U`, where `P` is a
/// permutation encoded by `piv`, `L` is unit lower-triangular and `U` is
/// upper-triangular. The two triangular factors share one matrix: `lu` holds
/// `U` on and above the diagonal and the `L` multipliers below it. At each
/// step the largest-magnitude entry in the pivot column is swapped to the
/// diagonal, which keeps the factorization stable for all but the most
/// ill-conditioned inputs.
pub struct LU {
  lu : Matrix
  piv : Array[Int]
  sign : Int
}

///|
/// Factor a square matrix with partial pivoting. Raises `Singular` when a
/// zero pivot is encountered.
pub fn lu(a : Matrix) -> LU raise LinAlgError {
  if !a.is_square() {
    raise DimensionMismatch
  }
  let n = a.rows()
  let u = a.copy()
  let piv = Array::make(n, 0)
  for i in 0..<n {
    piv[i] = i
  }
  let mut sign = 1
  for k in 0..<n {
    // Locate the pivot row: the largest |u[i,k]| at or below the diagonal.
    let mut p = k
    let mut best = u.get(k, k).abs()
    for i in (k + 1)..<n {
      let v = u.get(i, k).abs()
      if v > best {
        best = v
        p = i
      }
    }
    if best == 0.0 {
      raise Singular
    }
    if p != k {
      u.swap_rows(k, p)
      let t = piv[k]
      piv[k] = piv[p]
      piv[p] = t
      sign = -sign
    }
    // Eliminate below the diagonal; the multiplier is kept in u[i,k].
    let pivot_val = u.get(k, k)
    for i in (k + 1)..<n {
      let f = u.get(i, k) / pivot_val
      u.set(i, k, f)
      for j in (k + 1)..<n {
        u.set(i, j, u.get(i, j) - f * u.get(k, j))
      }
    }
  }
  { lu: u, piv, sign, }
}

///|
/// Solve `A·x = b` for the original matrix using the stored factorization.
pub fn LU::solve(self : LU, b : Vector) -> Vector raise LinAlgError {
  let n = self.lu.rows()
  if b.length() != n {
    raise DimensionMismatch
  }
  // Apply the row permutation: y_i = b[piv[i]].
  let x = Vector::zeros(n)
  for i in 0..<n {
    x.set(i, b.get(self.piv[i]))
  }
  // Forward substitution: L·y = P·b (L is unit lower-triangular).
  for i in 0..<n {
    let mut s = x.get(i)
    for j in 0..<i {
      s = s - self.lu.get(i, j) * x.get(j)
    }
    x.set(i, s)
  }
  // Back substitution: U·x = y.
  for i in 0..<n {
    let r = n - 1 - i
    let mut s = x.get(r)
    for j in (r + 1)..<n {
      s = s - self.lu.get(r, j) * x.get(j)
    }
    x.set(r, s / self.lu.get(r, r))
  }
  x
}

///|
/// Determinant of the original matrix (sign and product of the pivots).
pub fn LU::det(self : LU) -> Double {
  let n = self.lu.rows()
  let mut d = 1.0
  for i in 0..<n {
    d = d * self.lu.get(i, i)
  }
  if self.sign < 0 {
    -d
  } else {
    d
  }
}

///|
/// Inverse of the original matrix, computed column by column.
pub fn LU::inverse(self : LU) -> Matrix {
  let n = self.lu.rows()
  let inv = Matrix::zeros(n, n)
  for j in 0..<n {
    let e = Vector::zeros(n)
    e.set(j, 1.0)
    let col = self.solve(e) catch {
      _ => abort("LU::inverse: solve failed on a non-singular factorization")
    }
    for i in 0..<n {
      inv.set(i, j, col.get(i))
    }
  }
  inv
}

///|
/// Cholesky factorization of a symmetric positive-definite matrix.
///
/// `cholesky(a)` returns the lower-triangular `L` such that `A = L·Lᵀ`. Only
/// the lower triangle of `a` is read; symmetry is the caller's contract.
/// Raises `NotPositiveDefinite` when a diagonal pivot is non-positive.
pub fn cholesky(a : Matrix) -> Matrix raise LinAlgError {
  if !a.is_square() {
    raise DimensionMismatch
  }
  let n = a.rows()
  let l = Matrix::zeros(n, n)
  for i in 0..<n {
    for j in 0..<=i {
      let mut s = a.get(i, j)
      for k in 0..<j {
        s = s - l.get(i, k) * l.get(j, k)
      }
      if i == j {
        if s <= 0.0 {
          raise NotPositiveDefinite
        }
        l.set(i, j, s.sqrt())
      } else {
        l.set(i, j, s / l.get(j, j))
      }
    }
  }
  l
}

///|
/// Solve `A·x = b` for a symmetric positive-definite `A` via Cholesky.
pub fn cholesky_solve(a : Matrix, b : Vector) -> Vector raise LinAlgError {
  let n = a.rows()
  if b.length() != n {
    raise DimensionMismatch
  }
  let l = cholesky(a)
  // Forward substitution: L·y = b.
  let y = Vector::zeros(n)
  for i in 0..<n {
    let mut s = b.get(i)
    for j in 0..<i {
      s = s - l.get(i, j) * y.get(j)
    }
    y.set(i, s / l.get(i, i))
  }
  // Back substitution: Lᵀ·x = y.
  let x = Vector::zeros(n)
  for i in 0..<n {
    let r = n - 1 - i
    let mut s = y.get(r)
    for j in (r + 1)..<n {
      s = s - l.get(j, r) * x.get(j)
    }
    x.set(r, s / l.get(r, r))
  }
  x
}

///|
/// QR factorization of a tall (`m ≥ n`) matrix via Householder reflectors.
///
/// `qr(a)` returns `Q` (`m × m`, orthogonal) and `R` (`m × n`, upper
/// triangular in its top `n × n` block) such that `A = Q·R`. Raises
/// `DimensionMismatch` when the matrix has more columns than rows.
pub struct QR {
  q : Matrix
  r : Matrix
}

///|
pub fn qr(a : Matrix) -> QR raise LinAlgError {
  let m = a.rows()
  let n = a.cols()
  if m < n {
    raise DimensionMismatch
  }
  let r = a.copy()
  let q = Matrix::identity(m)
  for k in 0..<n {
    // Norm of the sub-column R[k:m, k].
    let mut normx = 0.0
    for i in k..<m {
      normx = normx + r.get(i, k) * r.get(i, k)
    }
    normx = normx.sqrt()
    if normx != 0.0 {
      let x0 = r.get(k, k)
      let alpha = if x0 >= 0.0 { -normx } else { normx }
      // Householder vector v = x - alpha·e₁.
      let v = Array::make(m - k, 0.0)
      v[0] = x0 - alpha
      let mut beta = v[0] * v[0]
      for i in (k + 1)..<m {
        v[i - k] = r.get(i, k)
        beta = beta + v[i - k] * v[i - k]
      }
      if beta != 0.0 {
        // Apply H = I - 2·v·vᵀ/β to R[k:m, k:n].
        for j in k..<n {
          let mut s = 0.0
          for i in k..<m {
            s = s + v[i - k] * r.get(i, j)
          }
          let w = 2.0 * s / beta
          for i in k..<m {
            r.set(i, j, r.get(i, j) - w * v[i - k])
          }
        }
        // Accumulate Q = Q·H.
        for i in 0..<m {
          let mut s = 0.0
          for l in k..<m {
            s = s + q.get(i, l) * v[l - k]
          }
          let w = 2.0 * s / beta
          for l in k..<m {
            q.set(i, l, q.get(i, l) - w * v[l - k])
          }
        }
      }
      // The reflector collapses column k to alpha·e₁; enforce clean zeros.
      r.set(k, k, alpha)
      for i in (k + 1)..<m {
        r.set(i, k, 0.0)
      }
    }
  }
  { q, r, }
}

///|
/// Solve the least-squares problem `min ‖A·x − b‖₂` using `A = Q·R`.
///
/// Computes `x = R⁻¹ · (Qᵀ b)` on the top `n × n` block of `R`.
pub fn QR::solve(self : QR, b : Vector) -> Vector raise LinAlgError {
  let m = self.q.rows()
  let n = self.r.cols()
  if b.length() != m {
    raise DimensionMismatch
  }
  let c = Vector::zeros(n)
  for i in 0..<n {
    let mut s = 0.0
    for j in 0..<m {
      s = s + self.q.get(j, i) * b.get(j)
    }
    c.set(i, s)
  }
  let x = Vector::zeros(n)
  for i in 0..<n {
    let row = n - 1 - i
    let mut s = c.get(row)
    for j in (row + 1)..<n {
      s = s - self.r.get(row, j) * x.get(j)
    }
    x.set(row, s / self.r.get(row, row))
  }
  x
}
