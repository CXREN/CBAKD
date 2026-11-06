/// Dense vector of 64-bit floats.
///
/// A `Vector` is a fixed-length array of `Double` values. It keeps the same
/// ownership model as `Matrix`: callers provide the backing array, and
/// construction validates length.

pub struct Vector {
  len : Int
  data : Array[Double]
}

/// Build a vector from an array.
pub fn Vector::new(data : Array[Double]) -> Vector {
  { len: data.length(), data }
}

/// The all-zero vector of length `n`.
pub fn Vector::zeros(n : Int) -> Vector {
  { len: n, data: Array::make(n, 0.0) }
}

/// The vector of length `n` filled with a constant value.
pub fn Vector::filled(n : Int, value : Double) -> Vector {
  { len: n, data: Array::make(n, value) }
}

/// Number of elements.
pub fn Vector::length(self : Vector) -> Int {
  self.len
}

/// Element at index `i`.
pub fn Vector::get(self : Vector, i : Int) -> Double {
  self.data[i]
}

/// Set element at index `i`.
pub fn Vector::set(self : Vector, i : Int, value : Double) -> Unit {
  self.data[i] = value
}

/// Dot product of two equal-length vectors.
pub fn Vector::dot(self : Vector, other : Vector) -> Double raise LinAlgError {
  if self.len != other.len {
    raise DimensionMismatch
  }
  let mut s = 0.0
  for i in 0..<self.len {
    s = s + self.data[i] * other.data[i]
  }
  s
}

/// Euclidean (ℓ₂) norm.
pub fn Vector::norm(self : Vector) -> Double {
  let mut s = 0.0
  for i in 0..<self.len {
    s = s + self.data[i] * self.data[i]
  }
  s.sqrt()
}

/// Scale every element by a scalar.
pub fn Vector::scale(self : Vector, k : Double) -> Vector {
  let v = Vector::zeros(self.len)
  for i in 0..<self.len {
    v.data[i] = self.data[i] * k
  }
  v
}

/// Element-wise sum of two equal-length vectors.
pub fn Vector::add(self : Vector, other : Vector) -> Vector raise LinAlgError {
  if self.len != other.len {
    raise DimensionMismatch
  }
  let v = Vector::zeros(self.len)
  for i in 0..<self.len {
    v.data[i] = self.data[i] + other.data[i]
  }
  v
}

/// Element-wise difference of two equal-length vectors.
pub fn Vector::sub(self : Vector, other : Vector) -> Vector raise LinAlgError {
  if self.len != other.len {
    raise DimensionMismatch
  }
  let v = Vector::zeros(self.len)
  for i in 0..<self.len {
    v.data[i] = self.data[i] - other.data[i]
  }
  v
}

/// `y <- y + a · x` (axpy), in place on `self`.
pub fn Vector::axpy(self : Vector, a : Double, x : Vector) -> Unit raise LinAlgError {
  if self.len != x.len {
    raise DimensionMismatch
  }
  for i in 0..<self.len {
    self.data[i] = self.data[i] + a * x.data[i]
  }
}

/// Compact `[a, b, c]` rendering, for debugging and demos.
pub fn Vector::to_string(self : Vector) -> String {
  let mut s = "["
  for i in 0..<self.len {
    if i > 0 {
      s = s + ", "
    }
    s = s + self.data[i].to_string()
  }
  s + "]"
}
