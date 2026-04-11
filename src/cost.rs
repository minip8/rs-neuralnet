use crate::matrix::Matrix;
use num_traits::Float;
pub mod functions;

pub struct Cost<T> {
    forward: fn(&Matrix<T>, &Matrix<T>) -> Matrix<T>,
    backward: fn(Matrix<T>, &Matrix<T>) -> Matrix<T>,
    loss: fn(&Matrix<T>, &Matrix<T>) -> T,
    _marker: std::marker::PhantomData<T>,
}

impl<F> Cost<F>
where
    F: Float,
{
    pub fn new(
        forward: fn(&Matrix<F>, &Matrix<F>) -> Matrix<F>,
        backward: fn(Matrix<F>, &Matrix<F>) -> Matrix<F>,
        loss: fn(&Matrix<F>, &Matrix<F>) -> F,
    ) -> Self {
        Self {
            forward,
            backward,
            loss,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn mse() -> Self {
        Self {
            forward: functions::mses,
            backward: functions::mse_backward,
            loss: functions::mse,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F> Cost<F>
where
    F: Float,
{
    pub fn forward(&self) -> fn(&Matrix<F>, &Matrix<F>) -> Matrix<F> {
        self.forward
    }

    pub fn backward(&self) -> fn(Matrix<F>, &Matrix<F>) -> Matrix<F> {
        self.backward
    }

    pub fn loss(&self) -> fn(&Matrix<F>, &Matrix<F>) -> F {
        self.loss
    }
}
