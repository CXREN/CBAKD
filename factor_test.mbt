fn close(a : Double, b : Double) -> Bool {
  (a - b).abs() < 1.0e-9
}

fn mat_close(a : Matrix, b : Matrix) -> Bool {
  if a.rows() != b.rows() || a.cols() != b.cols() {
    false
  } else {
    let mut ok = true
    for i in 0..<a.rows() {
      for j in 0..<a.cols() {
        if !close(a.get(i, j), b.get(i, j)) {
          ok = false
        }
      }
    }
    ok
  }
}

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
