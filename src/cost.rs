use crate::matrix::Matrix;
use num_traits::Float;
pub mod functions;

pub struct Cost<T> {
    forward: fn(Matrix<T>, &Matrix<T>) -> Matrix<T>,
    backward: fn(Matrix<T>, &Matrix<T>) -> Matrix<T>,
    _marker: std::marker::PhantomData<T>,
}

impl<F> Cost<F>
where
    F: Float,
{
    pub fn new(
        forward: fn(Matrix<F>, &Matrix<F>) -> Matrix<F>,
        backward: fn(Matrix<F>, &Matrix<F>) -> Matrix<F>,
    ) -> Self {
        Self {
            forward,
            backward,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn mse() -> Self {
        Self {
            forward: functions::mse_forward,
            backward: functions::mse_backward,
            _marker: std::marker::PhantomData,
        }
    }
}
