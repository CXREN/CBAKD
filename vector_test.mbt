///|
test "dot product of integer-valued vectors is exact" {
  let a = @CBAKD.Vector::new([1.0, 2.0, 3.0])
  let b = @CBAKD.Vector::new([4.0, 5.0, 6.0])
  assert_true(a.dot(b) == 32.0)
}

///|
test "euclidean norm of a 3-4 vector is 5" {
  let v = @CBAKD.Vector::new([3.0, 4.0])
  assert_true(v.norm() == 5.0)
}

///|
test "scale, add and sub are element-wise" {
  let v = @CBAKD.Vector::new([1.0, 2.0]).scale(2.0)
  assert_true(v.get(0) == 2.0)
  assert_true(v.get(1) == 4.0)
  let s = @CBAKD.Vector::new([1.0, 2.0]).add(@CBAKD.Vector::new([3.0, 4.0]))
  assert_true(s.get(0) == 4.0)
  assert_true(s.get(1) == 6.0)
}

///|
test "axpy updates the vector in place" {
  let y = @CBAKD.Vector::new([1.0, 1.0])
  let x = @CBAKD.Vector::new([2.0, 3.0])
  y.axpy(2.0, x)
  assert_true(y.get(0) == 5.0)
  assert_true(y.get(1) == 7.0)
}

///|
test "to_string renders a compact list" {
  assert_true(@CBAKD.Vector::new([1.0, 2.5]).to_string() == "[1, 2.5]")
}
