use num_traits::Float;

use crate::{device::Device, matrix::ops::MatrixOps};

#[derive(Clone)]
pub struct Cuda;

#[derive(Clone)]
pub struct CudaMatrix<T> {
    rows: usize,
    cols: usize,
    ptr: *mut T,
}

impl<T: Clone> Device<T> for Cuda {
    type Matrix = CudaMatrix<T>;
}

impl<T> MatrixOps<T> for CudaMatrix<T> {
    fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        todo!()
    }

    fn fill(v: T, rows: usize, cols: usize) -> Self
    where
        T: Copy,
    {
        todo!()
    }

    fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Float,
    {
        todo!()
    }

    fn numel(&self) -> usize {
        return self.rows * self.cols;
    }

    fn get(&self, r: usize, c: usize) -> T {
        todo!()
    }

    fn set_(&mut self, r: usize, c: usize, v: T) -> &mut Self {
        todo!()
    }

    fn set_row_(&mut self, r: usize, values: &[T]) -> &mut Self
    where
        T: Copy,
    {
        todo!()
    }

    fn apply_<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T,
    {
        todo!()
    }

    fn transpose(&self) -> Self
    where
        T: Copy,
    {
        todo!()
    }

    fn add_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        todo!()
    }

    fn sub_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        todo!()
    }

    fn hadamard_(&mut self, other: &Self) -> &mut Self
    where
        T: Float,
    {
        todo!()
    }

    fn mul_(&mut self, scale: T) -> &mut Self
    where
        T: Float,
    {
        todo!()
    }

    fn dot(&self, other: &Self) -> T
    where
        T: Float,
    {
        todo!()
    }

    fn mat_mul(&self, other: &Self) -> Self
    where
        T: Float,
    {
        todo!()
    }

    fn col_sum_(&mut self) -> &mut Self
    where
        T: Float,
    {
        todo!()
    }

    fn max(&self) -> (usize, T)
    where
        T: Float,
    {
        todo!()
    }
}
