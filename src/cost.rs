use crate::matrix::Matrix;
use num_traits::Float;
pub mod functions;

pub struct Cost<T, F1, F2> {
    forward: F1,
    backward: F2,
    _marker: std::marker::PhantomData<T>,
}

impl<F, F1, F2> Cost<F, F1, F2>
where
    F: Float,
    F1: Fn(&Matrix<F>, &Matrix<F>) -> Matrix<F>,
    F2: Fn(&Matrix<F>, &Matrix<F>) -> Matrix<F>,
{
    pub fn new(forward: F1, backward: F2) -> Self {
        Self {
            forward,
            backward,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F> Cost<F, fn(Matrix<F>, &Matrix<F>) -> Matrix<F>, fn(Matrix<F>, &Matrix<F>) -> Matrix<F>>
where
    F: Float,
{
    pub fn mse() -> Self {
        Self {
            forward: functions::mse_forward,
            backward: functions::mse_backward,
            _marker: std::marker::PhantomData,
        }
    }
}
