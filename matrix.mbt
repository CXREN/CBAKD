/// Dense row-major matrix of 64-bit floats.
///
/// A `Matrix` is a fixed `rows × cols` grid of `Double` values stored in a
/// single flat array in row-major order (row `i`, column `j` lives at index
/// `i * cols + j`). The type is deliberately lightweight: callers own the
/// data array, and construction only validates shape.

pub struct Matrix {
  rows : Int
  cols : Int
  data : Array[Double]
}

/// Build a matrix from a row-major array, validating the shape.
pub fn Matrix::new(rows : Int, cols : Int, data : Array[Double]) -> Matrix {
  if rows < 0 || cols < 0 {
    abort("Matrix::new: negative dimension")
  }
  if data.length() != rows * cols {
    abort("Matrix::new: data length does not match rows × cols")
  }
  { rows, cols, data }
}

/// The `rows × cols` zero matrix.
pub fn Matrix::zeros(rows : Int, cols : Int) -> Matrix {
  { rows, cols, data: Array::make(rows * cols, 0.0) }
}

/// The `rows × cols` matrix filled with a constant value.
pub fn Matrix::filled(rows : Int, cols : Int, value : Double) -> Matrix {
  { rows, cols, data: Array::make(rows * cols, value) }
}

/// The `n × n` identity matrix.
pub fn Matrix::identity(n : Int) -> Matrix {
  let data = Array::make(n * n, 0.0)
  for i in 0..<n {
    data[i * n + i] = 1.0
  }
  { rows: n, cols: n, data }
}

/// Number of rows.
pub fn Matrix::rows(self : Matrix) -> Int {
  self.rows
}

/// Number of columns.
pub fn Matrix::cols(self : Matrix) -> Int {
  self.cols
}

/// Element at row `i`, column `j`.
pub fn Matrix::get(self : Matrix, i : Int, j : Int) -> Double {
  self.data[i * self.cols + j]
}

/// Set element at row `i`, column `j`.
pub fn Matrix::set(self : Matrix, i : Int, j : Int, value : Double) -> Unit {
  self.data[i * self.cols + j] = value
}

/// True when the matrix is square.
pub fn Matrix::is_square(self : Matrix) -> Bool {
  self.rows == self.cols
}

/// Transpose: the `cols × rows` matrix with rows and columns swapped.
pub fn Matrix::transpose(self : Matrix) -> Matrix {
  let t = Matrix::zeros(self.cols, self.rows)
  for i in 0..<self.rows {
    for j in 0..<self.cols {
      t.data[j * self.rows + i] = self.data[i * self.cols + j]
    }
  }
  t
}

/// Multiply every element by a scalar.
pub fn Matrix::scale(self : Matrix, k : Double) -> Matrix {
  let m = Matrix::zeros(self.rows, self.cols)
  for i in 0..<self.data.length() {
    m.data[i] = self.data[i] * k
  }
  m
}

/// Element-wise sum of two same-shaped matrices.
pub fn Matrix::add(self : Matrix, other : Matrix) -> Matrix raise LinAlgError {
  if self.rows != other.rows || self.cols != other.cols {
    raise DimensionMismatch
  }
  let m = Matrix::zeros(self.rows, self.cols)
  for i in 0..<self.data.length() {
    m.data[i] = self.data[i] + other.data[i]
  }
  m
}

/// Element-wise difference of two same-shaped matrices.
pub fn Matrix::sub(self : Matrix, other : Matrix) -> Matrix raise LinAlgError {
  if self.rows != other.rows || self.cols != other.cols {
    raise DimensionMismatch
  }
  let m = Matrix::zeros(self.rows, self.cols)
  for i in 0..<self.data.length() {
    m.data[i] = self.data[i] - other.data[i]
  }
  m
}

/// Matrix product `self × other`.
pub fn Matrix::mul(self : Matrix, other : Matrix) -> Matrix raise LinAlgError {
  if self.cols != other.rows {
    raise DimensionMismatch
  }
  let m = Matrix::zeros(self.rows, other.cols)
  for i in 0..<self.rows {
    for k in 0..<self.cols {
      let a = self.data[i * self.cols + k]
      for j in 0..<other.cols {
        m.data[i * other.cols + j] = m.data[i * other.cols + j] + a * other.data[k * other.cols + j]
      }
    }
  }
  m
}

/// Matrix–vector product `self × v` where `v` has length `cols`.
pub fn Matrix::mul_vec(self : Matrix, v : Vector) -> Vector raise LinAlgError {
  if self.cols != v.length() {
    raise DimensionMismatch
  }
  let out = Vector::zeros(self.rows)
  for i in 0..<self.rows {
    let mut s = 0.0
    for j in 0..<self.cols {
      s = s + self.data[i * self.cols + j] * v.get(j)
    }
    out.set(i, s)
  }
  out
}

/// Frobenius norm: the square root of the sum of squared entries.
pub fn Matrix::norm_fro(self : Matrix) -> Double {
  let mut s = 0.0
  for i in 0..<self.data.length() {
    s = s + self.data[i] * self.data[i]
  }
  s.sqrt()
}

/// Maximum-absolute-column-sum norm (the operator 1-norm).
pub fn Matrix::norm_1(self : Matrix) -> Double {
  let mut best = 0.0
  for j in 0..<self.cols {
    let mut s = 0.0
    for i in 0..<self.rows {
      s = s + self.data[i * self.cols + j].abs()
    }
    if s > best {
      best = s
    }
  }
  best
}

/// Maximum-absolute-row-sum norm (the operator ∞-norm).
pub fn Matrix::norm_inf(self : Matrix) -> Double {
  let mut best = 0.0
  for i in 0..<self.rows {
    let mut s = 0.0
    for j in 0..<self.cols {
      s = s + self.data[i * self.cols + j].abs()
    }
    if s > best {
      best = s
    }
  }
  best
}
