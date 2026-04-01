use num_traits::Float;

use crate::{cost::Cost, layer::Layer, matrix::Matrix};

pub struct Network<T> {
    layers: Vec<Layer<T>>,
    cost: Cost<T>,
    lr: T,
}

impl<F> Network<F>
where
    F: Float,
{
    pub fn new(layers: Vec<Layer<F>>, cost: Cost<F>, lr: F) -> Self {
        Self { layers, cost, lr }
    }

    pub fn forward(&mut self, mut input: Matrix<F>) -> Matrix<F> {
        for i in 0..self.layers.len() {
            input = self.layers[i].forward(&input);
        }
        input
    }

    pub fn backward(&mut self, mut dc_da: Matrix<F>) -> Matrix<F> {
        for i in self.layers.len() - 1..0 {
            dc_da = self.layers[i].backward(dc_da, self.lr);
        }
        dc_da
    }

    pub fn step(&mut self, x: Matrix<F>, y: &Matrix<F>) -> F {
        let output = self.forward(x);
        let loss = self.cost.loss()(&output, y);
        let mses_backward = self.cost.backward()(output, y);
        self.backward(mses_backward);
        loss
    }
}
