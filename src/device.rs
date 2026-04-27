use crate::matrix::ops::MatrixOps;

pub trait Device<T>: Clone + 'static {
    type Matrix: Clone + MatrixOps<T>;
}
