use std::{cell::RefCell, fmt::Display};

use num_traits::Float;

use crate::{
    device::Device,
    matrix::{
        ops::MatrixOps,
        row_iter::{RowIter, RowIterMut},
    },
};

use rand::{
    SeedableRng,
    rngs::{StdRng, SysRng},
};

use rand_distr::{Distribution, Normal, StandardNormal};

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::try_from_rng(&mut SysRng).unwrap());
}

#[derive(Clone)]
pub struct Cpu;

pub struct CpuMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl Device for Cpu {
    type Matrix<T> = CpuMatrix<T>;
}

impl<T> MatrixOps<T> for CpuMatrix<T> {
    fn numel(&self) -> usize {
        self.rows * self.cols
    }
}

// Constructors
impl<F: Float> CpuMatrix<F> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![F::zero(); rows * cols],
        }
    }

    pub fn normal(rows: usize, cols: usize, mean: F, std_dev: F) -> Self
    where
        StandardNormal: Distribution<F>,
    {
        let norm = Normal::new(mean, std_dev).unwrap();
        CpuMatrix {
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

impl<T> CpuMatrix<T> {
    pub fn from_vec1d(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Self { rows, cols, data }
    }
}

impl<T: Copy + Default> CpuMatrix<T> {
    pub fn defaults(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    pub fn from_vec2d(data2d: Vec<Vec<T>>) -> Self {
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

// QOL
impl<T: Copy> CpuMatrix<T> {
    pub fn rowcol_to_idx(&self, r: usize, c: usize) -> usize {
        let idx = r * self.cols + c;
        // self.assert_idx_ok(idx);
        idx
    }

    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[self.rowcol_to_idx(r, c)]
    }

    pub fn item(&self) -> T {
        self.data[0]
    }

    pub fn set_(&mut self, r: usize, c: usize, v: T) -> &mut Self {
        let idx = self.rowcol_to_idx(r, c);
        self.data[idx] = v;
        self
    }

    pub fn set_row_(&mut self, r: usize, values: &[T]) -> &mut Self {
        let start = self.rowcol_to_idx(r, 0);
        let end = self.rowcol_to_idx(r + 1, 0);
        for i in start..end {
            self.data[i] = values[i - start];
        }
        self
    }

    pub fn set_row(mut self, r: usize, values: &[T]) -> Self {
        self.set_row_(r, values);
        self
    }

    pub fn apply_<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(T) -> T,
    {
        self.data.iter_mut().for_each(|x| *x = f(*x));
        self
    }

    pub fn apply<F>(mut self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        self.apply_(f);
        self
    }
}

// Unary operations
impl<F: Float> CpuMatrix<F> {
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

impl<F: Float> CpuMatrix<F> {
    pub fn add_(&mut self, other: &Self) -> &mut Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
        self
    }

    pub fn add(mut self, other: &Self) -> Self {
        self.add_(other);
        self
    }

    pub fn add_row_to_all_rows(&self, other: &Self) -> Self {
        let mut res = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(i, j, self.get(i, j) + other.get(0, j));
            }
        }
        res
    }

    pub fn add_row_to_all_cols(&self, other: &Self) -> Self {
        let mut res = Self::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(i, j, self.get(i, j) + other.get(0, i));
            }
        }
        res
    }

    pub fn sub_(&mut self, other: &Self) -> &mut Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
        self
    }

    pub fn sub(mut self, other: &Self) -> Self {
        self.sub_(other);
        self
    }

    pub fn hadamard_(&mut self, other: &Self) -> &mut Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a * *b);
        self
    }

    pub fn hadamard(mut self, other: &Self) -> Self {
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
        let end = start + self.cols;
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
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
    }

    /// Takes two same dimensional matrices and dot products their rows together
    /// Returns a row vector of the dot products
    pub fn dot_rows(&self, other: &Self) -> Self {
        let mut data = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            data[i] = {
                let start = i * self.cols;
                let end = start + self.cols;
                self.data[start..end]
                    .iter()
                    .zip(other.data[start..end].iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
            }
        }
        Self::from_vec1d(1, self.rows, data)
    }

    /// Takes a N x M CpuMatrix and a 1 x M CpuMatrix and dot products the row vector
    /// with every row of the N x M CpuMatrix
    /// Returns a row vector of the dot products
    pub fn dot_rows_with_row(&self, other: &Self) -> Self {
        let mut data = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            data[i] = {
                let start = i * self.cols;
                let end = start + self.cols;
                self.data[start..end]
                    .iter()
                    .zip(other.data.iter())
                    .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
            }
        }
        Self::from_vec1d(1, self.rows, data)
    }

    pub fn mat_mul_(&mut self, other: &Self) -> &mut Self {
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
        self.mat_mul_(other);
        self
    }

    /// self is a 1 x M CpuMatrix
    /// other is a N x M CpuMatrix
    /// returns a 1 x N row vector A, where A[0, i] is given by MSE(self[0], other[i])
    pub fn mses(&self, other: &Self) -> Self {
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

impl<F: Float> CpuMatrix<F> {
    pub fn col_sum_(&mut self) -> &mut Self {
        for i in 1..self.rows {
            for j in 0..self.cols {
                self.set_(0, j, self.get(0, j) + self.get(i, j));
            }
        }
        self.rows = 1;
        self.data.truncate(self.cols);
        self
    }

    pub fn col_sum(mut self) -> Self {
        self.col_sum_();
        self
    }

    pub fn max(&self) -> (usize, F) {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, a)| (i, *a))
            .unwrap()
    }
}

impl<T> CpuMatrix<T> {
    pub fn iter(&'_ self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn row_iter(&self) -> RowIter<'_, T> {
        RowIter::new(&self.data, self.rows, self.cols)
    }

    pub fn row_iter_mut(&mut self) -> RowIterMut<'_, T> {
        RowIterMut::new(self.data.as_mut_slice(), self.rows, self.cols)
    }
}

impl<T> IntoIterator for CpuMatrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<F: Float + Display> Display for CpuMatrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\ndims: ({}, {})\n", self.rows, self.cols).unwrap();
        for r in self.row_iter() {
            for x in r {
                write!(f, "{x:.2}, ").unwrap();
            }
            write!(f, "\n").unwrap();
        }
        write!(f, "\n").unwrap();
        Ok(())
    }
}
