use std::cell::RefCell;

use num_traits::Float;
use rand::{
    SeedableRng,
    rngs::{StdRng, SysRng},
};
use rand_distr::{Distribution, Normal, StandardNormal};

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::try_from_rng(&mut SysRng).unwrap());
}

#[derive(Clone)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

// Getters
impl<T> Matrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}

// Constructors
impl<T: Float> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    pub fn normal<F>(rows: usize, cols: usize, mean: F, std_dev: F) -> Matrix<F>
    where
        F: num_traits::Float,
        StandardNormal: Distribution<F>,
    {
        let norm = Normal::new(mean, std_dev).unwrap();
        Matrix {
            rows,
            cols,
            data: RNG.with(|rng| {
                norm.sample_iter(&mut *rng.borrow_mut())
                    .take(rows * cols)
                    .collect()
            }),
        }
    }
}

impl<T> Matrix<T> {
    pub fn from_vec1d(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Self::assert_rowcol_dimensions_match_data1d(rows, cols, &data);

        Self { rows, cols, data }
    }
}

impl<T: Copy + Default> Matrix<T> {
    pub fn defaults(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec2d(data2d: Vec<Vec<T>>) -> Self {
        Self::assert_rowcol_dimensions_match_data2d(&data2d);

        let rows = data2d.len();
        let cols = data2d[0].len();
        let mut res = Self::defaults(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.set_(i, j, data2d[i][j]);
            }
        }
        res
    }
}

// Assertions
impl<T> Matrix<T> {
    fn assert_rowcol_dimensions_match_data1d(rows: usize, cols: usize, data: &Vec<T>) {
        debug_assert_eq!(rows * cols, data.len());
    }

    fn assert_rowcol_dimensions_match_data2d(data2d: &Vec<Vec<T>>) {
        debug_assert!(data2d.len() > 0);
        debug_assert!(data2d.iter().all(|r| r.len() == data2d[0].len()));
    }

    fn assert_same_dimensions(&self, other: &Self) {
        debug_assert_eq!((self.rows, self.cols), (other.rows, other.cols));
    }

    fn assert_matmul_compatible(&self, other: &Self) {
        debug_assert_eq!(self.cols, other.rows);
    }

    fn assert_idx_ok(&self, idx: usize) {
        debug_assert!(idx < self.data.len());
    }

    fn assert_row_vector(m: &Self) {
        debug_assert!(m.rows == 1);
    }

    fn assert_row_add_compatible(&self, other: &Self) {
        Self::assert_row_vector(other);
        debug_assert_eq!(self.cols, other.cols);
    }

    fn assert_rowcol_add_compatible(&self, other: &Self) {
        Self::assert_row_vector(other);
        debug_assert_eq!(self.rows, other.cols);
    }

    fn assert_same_cols(&self, other: &Self) {
        debug_assert_eq!(self.cols, other.cols);
    }
}

// QOL
impl<T: Copy> Matrix<T> {
    pub fn rowcol_to_idx(&self, r: usize, c: usize) -> usize {
        let idx = r * self.cols + c;
        self.assert_idx_ok(idx);
        idx
    }

    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[self.rowcol_to_idx(r, c)]
    }

    pub fn set_(&mut self, r: usize, c: usize, v: T) -> &mut Self {
        let idx = self.rowcol_to_idx(r, c);
        self.data[idx] = v;
        self
    }

    pub fn apply_<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(T) -> T,
    {
        self.data.iter_mut().for_each(|x| *x = f(*x));
        self
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let data = self.data.iter().map(|&x| f(x)).collect::<Vec<_>>();
        Self::from_vec1d(self.rows, self.cols, data)
    }
}

// Unary operations
impl<F: Float> Matrix<F> {
    pub fn transpose(&self) -> Self {
        let mut res = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(j, i, self.get(i, j));
            }
        }
        res
    }
}

// Binary operations

impl<F: Float> Matrix<F> {
    pub fn add_(&mut self, other: &Self) -> &mut Self {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
        self
    }

    pub fn add(mut self, other: &Self) -> Self {
        self.assert_same_dimensions(other);

        self.add_(other);
        self
    }

    pub fn add_row_to_all_rows(&self, other: &Self) -> Self {
        Self::assert_row_add_compatible(&self, other);
        let mut res = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(i, j, res.get(i, j) + other.get(0, j));
            }
        }
        res
    }

    pub fn add_row_to_all_cols(&self, other: &Self) -> Self {
        Self::assert_rowcol_add_compatible(&self, other);
        let mut res = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(i, j, other.get(0, i));
            }
        }
        res
    }

    pub fn sub_(&mut self, other: &Self) -> &mut Self {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
        self
    }

    pub fn sub(mut self, other: &Self) -> Self {
        self.assert_same_dimensions(other);

        self.sub_(other);
        self
    }

    pub fn hadamard_(&mut self, other: &Self) -> &mut Self {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a * *b);
        self
    }

    pub fn hadamard(mut self, other: &Self) -> Self {
        self.assert_same_dimensions(other);

        self.hadamard_(other);
        self
    }

    pub fn mul_(&mut self, scale: F) -> &mut Self {
        self.data.iter_mut().for_each(|a| *a = *a * scale);
        self
    }

    pub fn mul(mut self, scale: F) -> Self {
        self.mul_(scale);
        self
    }

    pub fn mul_row_(&mut self, row: usize, scale: F) -> &mut Self {
        let start = row * self.cols;
        let end = row * (self.cols + 1);
        for i in start..end {
            self.data[i] = self.data[i] * scale;
        }
        self
    }

    pub fn mul_row(mut self, row: usize, scale: F) -> Self {
        self.mul_row_(row, scale);
        self
    }

    pub fn dot(&self, other: &Self) -> F {
        self.assert_same_dimensions(other);
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Takes two same dimensional matrices and dot products their rows together
    /// Returns a row vector of the dot products
    pub fn dot_rows(&self, other: &Self) -> Self {
        self.assert_same_dimensions(other);

        let mut data = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            data[i] = {
                let start = i * self.cols;
                let end = i * (self.cols + 1);
                self.data[start..end]
                    .iter()
                    .zip(other.data[start..end].iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
            }
        }
        Self::from_vec1d(1, self.rows, data)
    }

    /// Takes a N x M matrix and a 1 x M matrix and dot products the row vector
    /// with every row of the N x M matrix
    /// Returns a row vector of the dot products
    pub fn dot_rows_with_row(&self, other: &Self) -> Self {
        self.assert_same_cols(other);

        let mut data = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            data[i] = {
                let start = i * self.cols;
                let end = i * (self.cols + 1);
                self.data[start..end]
                    .iter()
                    .zip(other.data.iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
            }
        }
        Self::from_vec1d(1, self.rows, data)
    }

    pub fn mat_mul_(&mut self, other: &Self) -> &mut Self {
        self.assert_matmul_compatible(other);

        let mut data = vec![F::zero(); self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.cols + j] =
                        data[i * other.cols + j] + self.get(i, k) * other.get(k, j);
                }
            }
        }
        self.cols = other.cols;
        self.data = data;
        self
    }

    pub fn mat_mul(mut self, other: &Self) -> Self {
        self.assert_matmul_compatible(other);

        self.mat_mul_(other);
        self
    }

    /// self is a 1 x M matrix
    /// other is a N x M matrix
    /// returns a row vector A, where A[0, i] is given by MSE(self, other[i])
    fn mse(&self, other: &Self) -> Self {
        self.assert_same_cols(other);
        let mut res = Self::zeros(1, other.rows);

        for i in 0..other.rows {
            let mse = {
                let mut accum = F::zero();
                for j in 0..other.cols {
                    accum = accum + (self.get(0, j) - other.get(i, j)).powf(F::from(2).unwrap());
                }
                accum
            };
            res.set_(0, i, mse.div(F::from(other.cols).unwrap()));
        }
        res
    }
}

impl<T> Matrix<T> {
    pub fn iter(&'_ self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_f32() {
        let m: Matrix<f32> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matrix_f64() {
        let m: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matrix_add() {
        let m1: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2: Matrix<f64> = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.add(&m2);
        assert_eq!(result.data(), &vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_sub() {
        let m1: Matrix<f64> = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let m2: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m1.sub(&m2);
        assert_eq!(result.data(), &vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_matrix_hadamard() {
        let m1: Matrix<f64> = Matrix::from_vec2d(vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
        let m2: Matrix<f64> = Matrix::from_vec2d(vec![vec![2.0, 2.0], vec![3.0, 3.0]]);
        let result = m1.hadamard(&m2);
        assert_eq!(result.data(), &vec![4.0, 6.0, 12.0, 15.0]);
    }

    #[test]
    fn test_matrix_mul_scalar() {
        let m: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m.mul(2.0);
        assert_eq!(result.data(), &vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let m: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let result = m.transpose();
        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.data(), &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matrix_mat_mul() {
        let m1: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2: Matrix<f64> = Matrix::from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.mat_mul(&m2);
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(result.data(), &vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matrix_zeros() {
        let m: Matrix<f32> = Matrix::zeros(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data(), &vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matrix_from_vec1d() {
        let m: Matrix<f64> = Matrix::from_vec1d(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matrix_apply() {
        let mut m: Matrix<f64> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // Double each element
        m.apply_(|x| 2.0 * x);
        assert_eq!(m.data(), &vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_apply_add_constant() {
        let mut m: Matrix<f32> = Matrix::from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // Add 10 to each element
        m.apply_(|x| x + 10.0);
        assert_eq!(m.data(), &vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_matrix_normal_dimensions() {
        let m = Matrix::<f64>::normal(3, 4, 0.0, 1.0);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.data().len(), 12);
    }

    #[test]
    fn test_matrix_normal_values_vary() {
        let m = Matrix::<f32>::normal(10, 10, 0.0, 1.0);
        // Check that not all values are the same (extremely unlikely with random normal distribution)
        let first_value = m.data()[0];
        let all_same = m.data().iter().all(|&x| x == first_value);
        assert!(!all_same, "All values should not be identical");
    }

    #[test]
    fn test_matrix_normal_statistical_properties() {
        // Create a large matrix to test statistical properties
        let mean = 5.0;
        let std_dev = 2.0;
        let m = Matrix::<f64>::normal(100, 100, mean, std_dev);

        // Calculate sample mean
        let sample_mean = m.data().iter().sum::<f64>() / m.data().len() as f64;

        // Calculate sample standard deviation
        let variance = m
            .data()
            .iter()
            .map(|&x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / m.data().len() as f64;
        let sample_std_dev = variance.sqrt();

        // With 10000 samples, the sample mean should be close to the true mean
        // Using a generous tolerance for randomness
        assert!(
            (sample_mean - mean).abs() < 0.5,
            "Sample mean {} should be close to {}",
            sample_mean,
            mean
        );
        assert!(
            (sample_std_dev - std_dev).abs() < 0.5,
            "Sample std_dev {} should be close to {}",
            sample_std_dev,
            std_dev
        );
    }

    #[test]
    fn test_matrix_mse_basic() {
        // self is 1 x 3 (row vector)
        let self_vec = Matrix::<f64>::from_vec2d(vec![vec![1.0, 2.0, 3.0]]);
        // other is 2 x 3 (two row vectors)
        let other = Matrix::<f64>::from_vec2d(vec![
            vec![1.0, 2.0, 3.0], // MSE with self should be 0
            vec![2.0, 3.0, 4.0], // MSE with self should be (1^2 + 1^2 + 1^2) / 3 = 1.0
        ]);

        let result = self_vec.mse(&other);

        // Result should be 1 x 2 (one MSE value per row of other)
        assert_eq!(result.rows(), 1, "Result should have 1 row");
        assert_eq!(
            result.cols(),
            2,
            "Result should have 2 columns (one per row of other)"
        );

        // Check MSE values
        assert_eq!(
            result.get(0, 0),
            0.0,
            "MSE of identical vectors should be 0"
        );
        assert_eq!(result.get(0, 1), 1.0, "MSE should be 1.0");
    }

    #[test]
    fn test_matrix_mse_dimensions() {
        // Test with different dimensions
        let self_vec = Matrix::<f32>::from_vec2d(vec![vec![0.0, 0.0, 0.0, 0.0]]);
        let other = Matrix::<f32>::from_vec2d(vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ]);

        let result = self_vec.mse(&other);

        // Result should be 1 x 3 (one MSE per row of other)
        assert_eq!(result.rows(), 1);
        assert_eq!(result.cols(), 3);

        // MSE(0, [1,1,1,1]) = (1+1+1+1)/4 = 1.0
        // MSE(0, [2,2,2,2]) = (4+4+4+4)/4 = 4.0
        // MSE(0, [3,3,3,3]) = (9+9+9+9)/4 = 9.0
        assert_eq!(result.get(0, 0), 1.0);
        assert_eq!(result.get(0, 1), 4.0);
        assert_eq!(result.get(0, 2), 9.0);
    }

    #[test]
    fn test_matrix_mse_single_element() {
        let self_vec = Matrix::<f64>::from_vec2d(vec![vec![5.0]]);
        let other = Matrix::<f64>::from_vec2d(vec![vec![5.0], vec![7.0]]);

        let result = self_vec.mse(&other);
        assert_eq!(result.rows(), 1);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.get(0, 0), 0.0); // (5-5)^2/1 = 0
        assert_eq!(result.get(0, 1), 4.0); // (5-7)^2/1 = 4
    }
}
