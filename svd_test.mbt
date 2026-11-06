test "svd reconstructs A = U*diag(s)*Vᵀ" {
  let a = Matrix::new(4, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
  let (uu, s, vt) = svd(a)
  // U·diag(s)
  let us = Matrix::zeros(4, 3)
  for j in 0..<3 {
    for i in 0..<4 {
      us.set(i, j, uu.get(i, j) * s.get(j))
    }
  }
  assert_true(mat_close(us.mul(vt), a))
  // U has orthonormal columns.
  assert_true(mat_close(uu.transpose().mul(uu), Matrix::identity(3)))
  // Vᵀ is orthogonal.
  assert_true(mat_close(vt.mul(vt.transpose()), Matrix::identity(3)))
}

test "svd returns singular values in descending order" {
  let a = Matrix::new(3, 3, [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
  let (_, s, _) = svd(a)
  assert_true(close(s.get(0), 3.0))
  assert_true(close(s.get(1), 2.0))
  assert_true(close(s.get(2), 1.0))
}

test "svd of a rank-one matrix has one non-zero singular value" {
  // Outer product [1,2,3,4]ᵀ · [1,1,1] scaled.
  let a = Matrix::new(4, 3, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
  let (_, s, _) = svd(a)
  // Singular value = ||[1,2,3,4]|| * ||[1,1,1]|| = sqrt(30) * sqrt(3).
  let expected = 30.0.sqrt() * 3.0.sqrt()
  assert_true(close(s.get(0), expected))
  assert_true(close(s.get(1), 0.0))
  assert_true(close(s.get(2), 0.0))
}

test "svd raises DimensionMismatch when m < n" {
  let a = Matrix::zeros(2, 3)
  let is_dim = try {
    let _ = svd(a)
    false
  } catch {
    DimensionMismatch => true
    _ => false
  }
  assert_true(is_dim)
}
