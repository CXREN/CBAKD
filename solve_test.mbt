///|
test "solve solves a square system" {
  let a = @CBAKD.Matrix::new(2, 2, [3.0, 1.0, 1.0, 2.0])
  let b = @CBAKD.Vector::new([5.0, 5.0])
  let x = @CBAKD.solve(a, b)
  assert_true(close(x.get(0), 1.0))
  assert_true(close(x.get(1), 2.0))
}

///|
test "lstsq fits an overdetermined system" {
  let a = @CBAKD.Matrix::new(3, 1, [1.0, 1.0, 1.0])
  let b = @CBAKD.Vector::new([2.0, 3.0, 4.0])
  let x = @CBAKD.lstsq(a, b)
  // Least-squares fit of a constant is the mean.
  assert_true(close(x.get(0), 3.0))
}
