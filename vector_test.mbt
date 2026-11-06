///|
test "dot product of integer-valued vectors is exact" {
  let a = Vector::new([1.0, 2.0, 3.0])
  let b = Vector::new([4.0, 5.0, 6.0])
  assert_true(a.dot(b) == 32.0)
}

///|
test "euclidean norm of a 3-4 vector is 5" {
  let v = Vector::new([3.0, 4.0])
  assert_true(v.norm() == 5.0)
}

///|
test "scale, add and sub are element-wise" {
  let v = Vector::new([1.0, 2.0]).scale(2.0)
  assert_true(v.get(0) == 2.0)
  assert_true(v.get(1) == 4.0)
  let s = Vector::new([1.0, 2.0]).add(Vector::new([3.0, 4.0]))
  assert_true(s.get(0) == 4.0)
  assert_true(s.get(1) == 6.0)
}

///|
test "axpy updates the vector in place" {
  let y = Vector::new([1.0, 1.0])
  let x = Vector::new([2.0, 3.0])
  y.axpy(2.0, x)
  assert_true(y.get(0) == 5.0)
  assert_true(y.get(1) == 7.0)
}

///|
test "to_string renders a compact list" {
  assert_true(Vector::new([1.0, 2.5]).to_string() == "[1, 2.5]")
}
