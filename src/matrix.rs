use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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
impl<T: Copy + Default> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec1d(rows: usize, cols: usize, data: Vec<T>) -> Matrix<T> {
        Matrix::assert_rowcol_dimensions_match_data(rows, cols, &data);

        Matrix { rows, cols, data }
    }

    pub fn from_vec2d(data2d: Vec<Vec<T>>) -> Matrix<T> {
        debug_assert!(data2d.len() > 0);

        let rows = data2d.len();
        let cols = data2d[0].len();
        let mut res = Matrix::zeros(rows, cols);
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
    fn assert_rowcol_dimensions_match_data(rows: usize, cols: usize, data: &Vec<T>) {
        debug_assert_eq!(rows * cols, data.len())
    }

    fn assert_same_dimensions(&self, other: &Matrix<T>) {
        debug_assert_eq!((self.rows, self.cols), (other.rows, other.cols));
    }

    fn assert_matmul_compatible(&self, other: &Matrix<T>) {
        debug_assert_eq!(self.cols, other.rows);
    }

    fn assert_idx_ok(&self, idx: usize) {
        debug_assert!(idx < self.data.len());
    }
}

// QOL
impl<T: Copy> Matrix<T> {
    fn rowcol_to_idx(&self, r: usize, c: usize) -> usize {
        let idx = r * self.cols + c;
        self.assert_idx_ok(idx);
        idx
    }

    fn get(&self, r: usize, c: usize) -> T {
        self.data[self.rowcol_to_idx(r, c)]
    }

    fn set_(&mut self, r: usize, c: usize, v: T) {
        let idx = self.rowcol_to_idx(r, c);
        self.data[idx] = v;
    }
}

// Unary operations
impl<T: Copy + Default> Matrix<T> {
    pub fn transpose(&self) -> Matrix<T> {
        let mut res = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(j, i, self.get(i, j));
            }
        }
        res
    }
}

// Binary operations

impl<T: Copy + Default + Add<Output = T> + AddAssign> Matrix<T> {
    pub fn add_(&mut self, other: &Matrix<T>) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a += *b);
    }

    pub fn add(&self, other: &Matrix<T>) -> Matrix<T> {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }
}

impl<T: Copy + Default + Sub<Output = T> + SubAssign> Matrix<T> {
    pub fn sub_(&mut self, other: &Matrix<T>) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a -= *b);
    }

    pub fn sub(&self, other: &Matrix<T>) -> Matrix<T> {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a - *b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }
}

impl<T: Copy + Default + Mul<Output = T> + MulAssign> Matrix<T> {
    pub fn dot_(&mut self, other: &Matrix<T>) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a *= *b);
    }

    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }

    pub fn mul_(&mut self, scale: T) {
        self.data.iter_mut().for_each(|a| *a *= scale);
    }

    pub fn mul(&self, scale: T) -> Matrix<T> {
        let data = self.data.iter().map(|a| *a * scale).collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }
}

impl<T: Copy + Default + Mul<Output = T> + Add<Output = T> + AddAssign> Matrix<T> {
    pub fn mat_mul_(&mut self, other: &Matrix<T>) {
        self.assert_matmul_compatible(other);

        let mut data = vec![T::default(); self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.cols + j] += self.get(i, k) * other.get(k, j);
                }
            }
        }
        self.cols = other.cols;
        self.data = data;
    }

    pub fn mat_mul(&self, other: &Matrix<T>) -> Matrix<T> {
        self.assert_matmul_compatible(other);

        let mut data = vec![T::default(); self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.cols + j] += self.get(i, k) * other.get(k, j);
                }
            }
        }
        Matrix::from_vec1d(self.rows, other.cols, data)
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
    fn test_matrix_i32() {
        let m: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_matrix_i64() {
        let m: Matrix<i64> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_matrix_add() {
        let m1: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        let m2: Matrix<i32> = Matrix::from_vec2d(vec![vec![5, 6], vec![7, 8]]);
        let result = m1.add(&m2);
        assert_eq!(result.data(), &vec![6, 8, 10, 12]);
    }

    #[test]
    fn test_matrix_sub() {
        let m1: Matrix<i32> = Matrix::from_vec2d(vec![vec![5, 6], vec![7, 8]]);
        let m2: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        let result = m1.sub(&m2);
        assert_eq!(result.data(), &vec![4, 4, 4, 4]);
    }

    #[test]
    fn test_matrix_dot() {
        let m1: Matrix<i32> = Matrix::from_vec2d(vec![vec![2, 3], vec![4, 5]]);
        let m2: Matrix<i32> = Matrix::from_vec2d(vec![vec![2, 2], vec![3, 3]]);
        let result = m1.dot(&m2);
        assert_eq!(result.data(), &vec![4, 6, 12, 15]);
    }

    #[test]
    fn test_matrix_mul_scalar() {
        let m: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        let result = m.mul(2);
        assert_eq!(result.data(), &vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_matrix_transpose() {
        let m: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let result = m.transpose();
        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 2);
        assert_eq!(result.data(), &vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_matrix_mat_mul() {
        let m1: Matrix<i32> = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        let m2: Matrix<i32> = Matrix::from_vec2d(vec![vec![5, 6], vec![7, 8]]);
        let result = m1.mat_mul(&m2);
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(result.data(), &vec![19, 22, 43, 50]);
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
        let m: Matrix<i32> = Matrix::from_vec1d(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.data(), &vec![1, 2, 3, 4]);
    }
}
