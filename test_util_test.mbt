// Shared floating-point comparison helpers for the test suite.

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
