///|
test "identity is square with ones on the diagonal" {
  let i = @CBAKD.Matrix::identity(3)
  assert_true(i.rows() == 3)
  assert_true(i.cols() == 3)
  assert_true(i.is_square())
  for r in 0..<3 {
    for c in 0..<3 {
      if r == c {
        assert_true(i.get(r, c) == 1.0)
      } else {
        assert_true(i.get(r, c) == 0.0)
      }
    }
  }
}

///|
test "zeros and filled produce the expected shapes" {
  let z = @CBAKD.Matrix::zeros(2, 3)
  assert_true(z.rows() == 2)
  assert_true(z.cols() == 3)
  assert_true(z.get(1, 2) == 0.0)

  let f = @CBAKD.Matrix::filled(2, 2, 7.5)
  assert_true(f.get(0, 1) == 7.5)
}

///|
test "set writes through to the backing array" {
  let m = @CBAKD.Matrix::zeros(2, 2)
  m.set(0, 1, 4.0)
  assert_true(m.get(0, 1) == 4.0)
}

///|
test "transpose swaps rows and columns" {
  let m = @CBAKD.Matrix::new(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let t = m.transpose()
  assert_true(t.rows() == 3)
  assert_true(t.cols() == 2)
  assert_true(t.get(0, 0) == 1.0)
  assert_true(t.get(0, 1) == 4.0)
  assert_true(t.get(2, 1) == 6.0)
}

///|
test "matrix product of integer-valued matrices is exact" {
  let a = @CBAKD.Matrix::new(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let b = @CBAKD.Matrix::new(3, 2, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
  let c = a.mul(b)
  assert_true(c.rows() == 2)
  assert_true(c.cols() == 2)
  assert_true(c.get(0, 0) == 58.0) // 1*7 + 2*9 + 3*11
  assert_true(c.get(0, 1) == 64.0)
  assert_true(c.get(1, 0) == 139.0)
  assert_true(c.get(1, 1) == 154.0)
}

///|
test "add and sub are element-wise and reject mismatched shapes" {
  let a = @CBAKD.Matrix::new(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = @CBAKD.Matrix::new(2, 2, [10.0, 20.0, 30.0, 40.0])
  let s = a.add(b)
  assert_true(s.get(0, 0) == 11.0)
  assert_true(s.get(1, 1) == 44.0)
  let d = b.sub(a)
  assert_true(d.get(0, 1) == 18.0)
}

///|
test "scale multiplies every element" {
  let m = @CBAKD.Matrix::new(2, 2, [1.0, 2.0, 3.0, 4.0]).scale(3.0)
  assert_true(m.get(0, 0) == 3.0)
  assert_true(m.get(1, 1) == 12.0)
}

///|
test "frobenius norm of a 3-4-5 matrix is 5" {
  let m = @CBAKD.Matrix::new(1, 3, [3.0, 4.0, 0.0])
  assert_true(m.norm_fro() == 5.0)
}

///|
test "one and infinity norms match row/column sums" {
  let m = @CBAKD.Matrix::new(2, 2, [1.0, -3.0, 2.0, 5.0])
  // column sums: |1|+|2|=3, |-3|+|5|=8 -> norm_1 = 8
  assert_true(m.norm_1() == 8.0)
  // row sums: |1|+|-3|=4, |2|+|5|=7 -> norm_inf = 7
  assert_true(m.norm_inf() == 7.0)
}
