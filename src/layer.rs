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
            // 1 x output_size
            bias: Matrix::zeros(1, output_size),

            // batch_size x input_size
            input: Matrix::zeros(0, 0),

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
    /// a is a batch_size x input_size matrix
    /// a x weights is a batch_size x output_size matrix
    fn forward(&mut self, a: &Matrix<F>) -> Matrix<F> {
        self.input = a.clone();
        let res = a
            .clone()
            .mat_mul(&self.weights)
            .add_row_to_all_rows(&self.bias);
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
    fn backward(&mut self, dc_da: Matrix<F>, lr: F) -> Matrix<F> {
        let batch_size = self.input.rows();
        let batch_size_inv = F::one() / F::from(batch_size).unwrap();
        let inputs_transposed = self.input.clone().transpose();
        let da_dz = self
            .pre_activation
            .clone()
            .apply(|x| self.activation.backward(x));

        let delta = dc_da.clone().hadamard(&da_dz);

        let dc_dw = inputs_transposed.mat_mul(&delta);

        let dc_db = delta.clone().col_sum().mul(batch_size_inv);

        let dc_da_prev = delta.mat_mul(&self.weights.transpose());

        self.weights.sub_(&dc_dw.mul(lr));
        self.bias.sub_(&dc_db.mul(lr));

        dc_da_prev
    }
}
