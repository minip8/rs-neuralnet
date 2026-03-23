use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use crate::{activation::Activation, matrix::Matrix};

pub struct Layer<T> {
    /// Note that input and output vectors are row vectors
    weights: Matrix<T>,
    bias: Matrix<T>,
    input: Matrix<T>,
    pre_activation: Matrix<T>,
    activation: Activation<T, fn(T) -> T, fn(T) -> T>,
}

impl<F> Layer<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::<F>::normal(
                input_size,
                output_size,
                F::zero(),
                F::sqrt(F::from(2).unwrap() / F::from(input_size).unwrap()),
            ),
            bias: Matrix::zeros(1, output_size),
            input: Matrix::zeros(1, input_size),
            pre_activation: Matrix::zeros(1, output_size),
            activation: Activation::relu(),
        }
    }
}
