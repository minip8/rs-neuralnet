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

#[derive(Clone)]
pub struct CpuMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Clone> Device<T> for Cpu {
    type Matrix = CpuMatrix<T>;
}

impl<T> MatrixOps<T> for CpuMatrix<T> {
    fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    fn fill(v: T, rows: usize, cols: usize) -> Self
    where
        T: Copy,
    {
        Self {
            data: vec![v; rows * cols],
            rows,
            cols,
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Float,
    {
        Self::fill(T::zero(), rows, cols)
    }

    fn numel(&self) -> usize {
        self.rows * self.cols
    }

    fn get(&self, r: usize, c: usize) -> T
    where
        T: Copy,
    {
        self.data[r * self.cols + c]
    }

    fn set_(&mut self, r: usize, c: usize, v: T) -> &mut Self {
        let idx = self.rowcol_to_idx(r, c);
        self.data[idx] = v;
        self
    }

    fn set_row_(&mut self, r: usize, values: &[T]) -> &mut Self
    where
        T: Copy,
    {
        let start = self.rowcol_to_idx(r, 0);
        let end = self.rowcol_to_idx(r + 1, 0);
        for i in start..end {
            self.data[i] = values[i - start];
        }
        self
    }

    fn apply_<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        self.data.iter().map(f);
        self
    }

    fn transpose(&self) -> Self
    where
        T: Copy,
    {
        let mut res = Self::fill(self.get(0, 0), self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(j, i, self.get(i, j));
            }
        }
        res
    }

    fn add_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a + *b);
        self
    }

    fn sub_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a - *b);
        self
    }

    fn hadamard_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a = *a * *b);
        self
    }

    fn mul_(&mut self, scale: T) -> &mut Self
    where
        T: Float,
    {
        self.data.iter_mut().for_each(|a| *a = *a * scale);
        self
    }

    fn dot(&self, other: &Self) -> T
    where
        T: Float,
    {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }

    fn mat_mul(&self, other: &Self) -> Self
    where
        T: Float,
    {
        let mut data = vec![T::zero(); self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.cols + j] =
                        data[i * other.cols + j] + self.get(i, k) * other.get(k, j);
                }
            }
        }
        Self::new(data, self.rows, other.cols)
    }

    fn col_sum_(&mut self) -> &mut Self
    where
        T: Float,
    {
        for i in 1..self.rows {
            for j in 0..self.cols {
                self.set_(0, j, self.get(0, j) + self.get(i, j));
            }
        }
        self.rows = 1;
        self.data.truncate(self.cols);
        self
    }

    fn max(&self) -> (usize, T)
    where
        T: Float,
    {
        self.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, a)| (i, *a))
            .unwrap()
    }
}

impl<T> CpuMatrix<T> {
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// QOL
impl<T> CpuMatrix<T> {
    pub fn rowcol_to_idx(&self, r: usize, c: usize) -> usize {
        let idx = r * self.cols + c;
        idx
    }
}

// Binary operations

impl<F: Float> CpuMatrix<F> {
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
        Self::new(data, 1, self.rows)
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
        Self::new(data, 1, self.rows)
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
