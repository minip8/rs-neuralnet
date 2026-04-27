use crate::matrix::ops::MatrixOps;

pub trait Device: Clone + 'static {
    type Matrix<T>: MatrixOps<T>;
}
