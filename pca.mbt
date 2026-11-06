/// Principal component analysis of a data matrix.
///
/// `pca(x)` treats rows of `x` as samples and columns as features. It centers
/// the features, forms the sample covariance matrix, and returns its
/// eigendecomposition sorted by descending variance. The result carries the
/// feature means, the principal components as orthonormal columns, and the
/// variances (eigenvalues) of each component.

pub struct PCA {
  mean : Vector
  components : Matrix
  variance : Vector
}

/// Fit a PCA model. Requires at least two samples (`m ≥ 2`).
pub fn pca(x : Matrix) -> PCA raise LinAlgError {
  let m = x.rows()
  let n = x.cols()
  if m < 2 {
    raise DimensionMismatch
  }
  // Column means.
  let mean = Vector::zeros(n)
  for j in 0..<n {
    let mut s = 0.0
    for i in 0..<m {
      s = s + x.get(i, j)
    }
    mean.set(j, s / m.to_double())
  }
  // Center the data.
  let xc = Matrix::zeros(m, n)
  for i in 0..<m {
    for j in 0..<n {
      xc.set(i, j, x.get(i, j) - mean.get(j))
    }
  }
  // Sample covariance = Xcᵀ·Xc / (m - 1).
  let cov = xc.transpose().mul(xc)
  let scale = 1.0 / (m - 1).to_double()
  let (values, vectors) = eig_sym(cov.scale(scale))
  // Sort components by descending variance.
  let idx = Array::make(n, 0)
  for i in 0..<n {
    idx[i] = i
  }
  for i in 0..<n {
    let mut best = i
    for k in (i + 1)..<n {
      if values.get(idx[k]) > values.get(idx[best]) {
        best = k
      }
    }
    let t = idx[i]
    idx[i] = idx[best]
    idx[best] = t
  }
  let variance = Vector::zeros(n)
  let components = Matrix::zeros(n, n)
  for j in 0..<n {
    let src = idx[j]
    variance.set(j, values.get(src))
    for i in 0..<n {
      components.set(i, j, vectors.get(i, src))
    }
  }
  { mean, components, variance }
}

/// Center new data with the learned means and project it onto the components.
pub fn PCA::transform(self : PCA, x : Matrix) -> Matrix raise LinAlgError {
  let m = x.rows()
  let n = x.cols()
  if n != self.mean.length() {
    raise DimensionMismatch
  }
  let xc = Matrix::zeros(m, n)
  for i in 0..<m {
    for j in 0..<n {
      xc.set(i, j, x.get(i, j) - self.mean.get(j))
    }
  }
  xc.mul(self.components)
}
