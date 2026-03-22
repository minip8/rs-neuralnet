use crate::{activation::Activation, matrix::Matrix};

pub struct Layer<T> {
    weights: Matrix<T>,
    bias: Matrix<T>,
    input: Matrix<T>,
    pre_activation: Matrix<T>,
    activation: Activation<T, fn(T) -> T, fn(T) -> T>,
}
