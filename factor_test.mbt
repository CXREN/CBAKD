fn lu_lower(l : LU) -> Matrix {
  let n = l.lu.rows()
  let m = Matrix::identity(n)
  for i in 0..<n {
    for j in 0..<i {
      m.set(i, j, l.lu.get(i, j))
    }
  }
  m
}

fn lu_upper(l : LU) -> Matrix {
  let n = l.lu.rows()
  let m = Matrix::zeros(n, n)
  for i in 0..<n {
    for j in i..<n {
      m.set(i, j, l.lu.get(i, j))
    }
  }
  m
}

test "lu reconstructs P*A = L*U with partial pivoting" {
  let a = Matrix::new(3, 3, [0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0, 3.0, 0.0])
  let f = lu(a)
  let n = 3
  let pa = Matrix::zeros(n, n)
  for i in 0..<n {
    for j in 0..<n {
      pa.set(i, j, a.get(f.piv[i], j))
    }
  }
  let lu_product = lu_lower(f).mul(lu_upper(f))
  assert_true(mat_close(pa, lu_product))
}

test "lu solves a linear system" {
  let a = Matrix::new(3, 3, [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0])
  let b = Vector::new([8.0, -11.0, -3.0])
  let x = lu(a).solve(b)
  // Known solution x = [2, 3, -1].
  assert_true(close(x.get(0), 2.0))
  assert_true(close(x.get(1), 3.0))
  assert_true(close(x.get(2), -1.0))
}

test "lu computes the determinant" {
  let a = Matrix::new(2, 2, [1.0, 2.0, 3.0, 4.0])
  assert_true(close(lu(a).det(), -2.0))
}

test "lu inverts an invertible matrix" {
  let a = Matrix::new(2, 2, [4.0, 7.0, 2.0, 6.0])
  let inv = lu(a).inverse()
  assert_true(mat_close(a.mul(inv), Matrix::identity(2)))
}

test "lu raises Singular on a singular matrix" {
  let a = Matrix::new(2, 2, [1.0, 2.0, 2.0, 4.0])
  let is_singular = try {
    let _ = lu(a)
    false
  } catch {
    Singular => true
    _ => false
  }
  assert_true(is_singular)
}

test "lu raises DimensionMismatch on a non-square matrix" {
  let a = Matrix::zeros(2, 3)
  let is_dim = try {
    let _ = lu(a)
    false
  } catch {
    DimensionMismatch => true
    _ => false
  }
  assert_true(is_dim)
}

test "cholesky reconstructs A = L*Lᵀ" {
  let a = Matrix::new(3, 3, [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0])
  let l = cholesky(a)
  assert_true(close(l.get(0, 0), 2.0))
  assert_true(close(l.get(1, 0), 6.0))
  assert_true(close(l.get(1, 1), 1.0))
  assert_true(close(l.get(2, 0), -8.0))
  assert_true(close(l.get(2, 1), 5.0))
  assert_true(close(l.get(2, 2), 3.0))
  assert_true(mat_close(l.mul(l.transpose()), a))
}

test "cholesky_solve solves a positive-definite system" {
  let a = Matrix::new(3, 3, [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0])
  let b = Vector::new([-20.0, -43.0, 192.0])
  let x = cholesky_solve(a, b)
  // Known solution x = [1, 2, 3].
  assert_true(close(x.get(0), 1.0))
  assert_true(close(x.get(1), 2.0))
  assert_true(close(x.get(2), 3.0))
}

test "cholesky raises NotPositiveDefinite on an indefinite matrix" {
  let a = Matrix::new(2, 2, [1.0, 2.0, 2.0, 1.0])
  let is_npd = try {
    let _ = cholesky(a)
    false
  } catch {
    NotPositiveDefinite => true
    _ => false
  }
  assert_true(is_npd)
}

test "qr reconstructs A = Q*R and Q is orthogonal" {
  let a = Matrix::new(4, 3, [1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 3.0, 9.0, 1.0, 4.0, 16.0])
  let f = qr(a)
  assert_true(mat_close(f.q.mul(f.r), a))
  assert_true(mat_close(f.q.transpose().mul(f.q), Matrix::identity(4)))
}

test "qr solves an overdetermined least-squares system" {
  // Fit y = 2 + 3x at x = 0,1,2,3.
  let a = Matrix::new(4, 2, [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
  let b = Vector::new([2.0, 5.0, 8.0, 11.0])
  let x = qr(a).solve(b)
  assert_true(close(x.get(0), 2.0))
  assert_true(close(x.get(1), 3.0))
}

test "qr raises DimensionMismatch when m < n" {
  let a = Matrix::zeros(2, 3)
  let is_dim = try {
    let _ = qr(a)
    false
  } catch {
    DimensionMismatch => true
    _ => false
  }
  assert_true(is_dim)
}
