/// Runnable demo of CBAKD: a tour of the factorizations, solvers and PCA
/// over a handful of small, hand-checkable problems.
///
/// Run with: `moon run example`

fn main raise {
  println("CBAKD — dense numerical linear algebra (demo)")
  println("")

  // 1. Linear system via LU with partial pivoting.
  let a = @cbakd.Matrix::new(2, 2, [3.0, 1.0, 1.0, 2.0])
  let b = @cbakd.Vector::new([5.0, 5.0])
  let x = @cbakd.solve(a, b)
  println("solve  [3 1; 1 2] x = [5 5]   ->  x = [\{x.get(0)}, \{x.get(1)}]")

  // 2. Least-squares line fit via Householder QR.
  let design = @cbakd.Matrix::new(4, 2, [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
  let y = @cbakd.Vector::new([2.0, 5.0, 8.0, 11.0])
  let fit = @cbakd.lstsq(design, y)
  println("lstsq  fit y = 2 + 3x        ->  intercept = \{fit.get(0)}, slope = \{fit.get(1)}")

  // 3. Cholesky factor of a symmetric positive-definite matrix.
  let spd = @cbakd.Matrix::new(3, 3, [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0])
  let l = @cbakd.cholesky(spd)
  let ll = l.mul(l.transpose())
  println("cholesky  L·Lᵀ diagonal      ->  [\{ll.get(0, 0)}, \{ll.get(1, 1)}, \{ll.get(2, 2)}]")

  // 4. Symmetric eigendecomposition via Jacobi rotations.
  let sym = @cbakd.Matrix::new(2, 2, [2.0, 1.0, 1.0, 2.0])
  let (w, _) = @cbakd.eig_sym(sym)
  println("eig_sym  [2 1; 1 2]          ->  eigenvalues {\{w.get(0)}, \{w.get(1)}}")

  // 5. Singular value decomposition via one-sided Jacobi.
  let m = @cbakd.Matrix::new(4, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  let (_, s, _) = @cbakd.svd(m)
  println("svd      singular values      ->  [\{s.get(0)}, \{s.get(1)}]")

  // 6. Principal component analysis of perfectly correlated data.
  let data = @cbakd.Matrix::new(4, 2, [1.0, 3.0, 2.0, 6.0, 3.0, 9.0, 4.0, 12.0])
  let p = @cbakd.pca(data)
  println("pca      variance             ->  [\{p.variance.get(0)}, \{p.variance.get(1)}]")
  println("pca      first component      ->  [\{p.components.get(0, 0)}, \{p.components.get(1, 0)}]")
}
