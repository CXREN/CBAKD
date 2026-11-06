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
  { lu: u, piv, sign }
}

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

/// Inverse of the original matrix, computed column by column.
pub fn LU::inverse(self : LU) -> Matrix {
  let n = self.lu.rows()
  let inv = Matrix::zeros(n, n)
  for j in 0..<n {
    let e = Vector::zeros(n)
    e.set(j, 1.0)
    let col = try {
      self.solve(e)
    } catch {
      _ => abort("LU::inverse: solve failed on a non-singular factorization")
    }
    for i in 0..<n {
      inv.set(i, j, col.get(i))
    }
  }
  inv
}
