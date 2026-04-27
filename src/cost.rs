use crate::{
    device::Device,
    matrix::{Matrix, cpu::Cpu},
};
use num_traits::Float;
pub mod functions;

pub struct Cost<T, D: Device<T>> {
    forward: fn(&Matrix<T, D>, &Matrix<T, D>) -> Matrix<T, D>,
    backward: fn(Matrix<T, D>, &Matrix<T, D>) -> Matrix<T, D>,
    loss: fn(&Matrix<T, D>, &Matrix<T, D>) -> T,
    // _marker: std::marker::PhantomData<T>,
}

impl<F, D: Device<F>> Cost<F, D>
where
    F: Float,
{
    pub fn new(
        forward: fn(&Matrix<F, D>, &Matrix<F, D>) -> Matrix<F, D>,
        backward: fn(Matrix<F, D>, &Matrix<F, D>) -> Matrix<F, D>,
        loss: fn(&Matrix<F, D>, &Matrix<F, D>) -> F,
    ) -> Self {
        Self {
            forward,
            backward,
            loss,
            // _marker: std::marker::PhantomData,
        }
    }
}

impl<F> Cost<F, Cpu>
where
    F: Float,
{
    pub fn mse() -> Self {
        Self::new(functions::mses, functions::mse_backward, functions::mse)
    }

    pub fn cross_entropy() -> Self {
        Self::new(
            functions::cross_entropy_loss_forward,
            functions::cross_entropy_loss_backward,
            functions::cross_entropy_loss,
        )
    }
}

impl<F, D: Device<F>> Cost<F, D>
where
    F: Float,
{
    pub fn forward(&self) -> fn(&Matrix<F, D>, &Matrix<F, D>) -> Matrix<F, D> {
        self.forward
    }

    pub fn backward(&self) -> fn(Matrix<F, D>, &Matrix<F, D>) -> Matrix<F, D> {
        self.backward
    }

    pub fn loss(&self) -> fn(&Matrix<F, D>, &Matrix<F, D>) -> F {
        self.loss
    }
}
