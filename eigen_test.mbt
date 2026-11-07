///|
test "eig_sym diagonalizes a symmetric matrix" {
  let a = @CBAKD.Matrix::new(3, 3, [4.0, 1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 0.0, 5.0])
  let (w, v) = @CBAKD.eig_sym(a)
  let n = 3
  // A·V = V·diag(w)
  let vw = @CBAKD.Matrix::zeros(n, n)
  for j in 0..<n {
    for i in 0..<n {
      vw.set(i, j, v.get(i, j) * w.get(j))
    }
  }
  assert_true(mat_close(a.mul(v), vw))
  // V is orthonormal: Vᵀ·V = I.
  assert_true(mat_close(v.transpose().mul(v), @CBAKD.Matrix::identity(n)))
}

///|
test "eig_sym returns known eigenvalues for a 2x2" {
  let a = @CBAKD.Matrix::new(2, 2, [2.0, 1.0, 1.0, 2.0])
  let (w, v) = @CBAKD.eig_sym(a)
  // Eigenvalues 3 and 1: trace 4, determinant 3.
  assert_true(close(w.get(0) + w.get(1), 4.0))
  assert_true(close(w.get(0) * w.get(1), 3.0))
  // Eigenvectors are orthonormal.
  assert_true(close(v.get(0, 0) * v.get(0, 0) + v.get(1, 0) * v.get(1, 0), 1.0))
}

///|
test "eig_sym handles a diagonal matrix" {
  let a = @CBAKD.Matrix::new(3, 3, [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0])
  let (w, v) = @CBAKD.eig_sym(a)
  let mut sum = 0.0
  for i in 0..<3 {
    sum = sum + w.get(i)
  }
  assert_true(close(sum, 6.0))
  // The eigenvectors are the coordinate axes, so V is a permutation/identity.
  assert_true(mat_close(v.transpose().mul(v), @CBAKD.Matrix::identity(3)))
}
