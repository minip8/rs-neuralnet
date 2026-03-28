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
    pub fn normal_relu(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix::<F>::normal(
                input_size,
                output_size,
                F::zero(),
                F::sqrt(F::from(2).unwrap() / F::from(input_size).unwrap()),
            ),
            // batch_size x input_size
            bias: Matrix::zeros(0, 0),

            // 1 x output_size
            input: Matrix::zeros(1, output_size),

            // batch_size x output_size
            pre_activation: Matrix::zeros(0, 0),
            activation: Activation::relu(),
        }
    }
}

impl<F> Layer<F>
where
    F: Float,
{
    fn forward(&mut self, a: &Matrix<F>) -> Matrix<F> {
        self.input = a.clone();
        let res = a.clone().mat_mul(&self.weights).add_row_to_all_rows(&self.bias);
        self.pre_activation = res.clone();

        res.apply(|x| self.activation.forward(x))
    }

    /// Let N be the number of neurons in the previous layer
    /// Let M be the number of neurons in this layer
    /// Takes in dc/da (derivative of c w.r.t this layer's activation)
    /// dc/da is a 1 x M row vector
    ///
    /// Computes dc/dw (derivative of c w.r.t this layer's weights) = dz/dw * da/dz * dc/da
    ///                                                             = a(L-1)* da/dz * dc/da
    /// Computes dc/db (derivative of c w.r.t this layer's bias)    = dz/db * da/dz * dc/da
    ///                                                             = 1     * da/dz * dc/da
    ///
    /// Returns (dc w.r.t previous layer's activation, dc w.r.t this layer's weights)
    fn backward(&mut self, dc_da: Matrix<F>) -> (Matrix<F>, Matrix<F>) {
        let da_dz = self.pre_activation.clone().apply(|x| self.activation.backward(x));
        let dc_da_prev = self.weights.dot_rows_with_row(&da_dz).hadamard(&dc_da);

        let dc_dw = {
            let m = da_dz.hadamard(&dc_da);
            let mut res = Matrix::<F>::zeros(self.input.rows(), self.pre_activation.cols());
            for i in 0..res.rows() {
                res.set_row_(i, m.data());
                res.mul_row_(i, self.input.get(0, i));
            }
            res
        };
        (dc_da_prev, dc_dw)
    }
}
