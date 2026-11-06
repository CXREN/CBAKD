test "pca finds the principal direction of perfectly correlated data" {
  let x = Matrix::new(4, 2, [1.0, 3.0, 2.0, 6.0, 3.0, 9.0, 4.0, 12.0])
  let p = pca(x)
  // One principal direction carries all variance: 50/3 and 0.
  assert_true(close(p.variance.get(0), 50.0 / 3.0))
  assert_true(close(p.variance.get(1), 0.0))
  // First component is a unit vector parallel to [1, 3].
  let c0 = p.components.get(0, 0)
  let c1 = p.components.get(1, 0)
  assert_true(close(c0 * c0 + c1 * c1, 1.0))
  assert_true(close((c1 / c0).abs(), 3.0))
  // Components are orthonormal.
  assert_true(mat_close(p.components.transpose().mul(p.components), Matrix::identity(2)))
}

test "pca transform centers and projects onto the principal axes" {
  let x = Matrix::new(4, 2, [1.0, 3.0, 2.0, 6.0, 3.0, 9.0, 4.0, 12.0])
  let p = pca(x)
  let scores = p.transform(x)
  // The minor axis has zero score everywhere.
  for i in 0..<4 {
    assert_true(close(scores.get(i, 1), 0.0))
  }
}

test "pca raises DimensionMismatch with fewer than two samples" {
  let x = Matrix::new(1, 2, [1.0, 2.0])
  let is_dim = try {
    let _ = pca(x)
    false
  } catch {
    DimensionMismatch => true
    _ => false
  }
  assert_true(is_dim)
}
