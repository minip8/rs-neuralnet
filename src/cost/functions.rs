use num_traits::Float;

use crate::matrix::Matrix;

/// xs: batch_size x output_size
/// y : 1 x output_size
/// Returns a 1 x batch_size row vector of the MSEs of all individual batches
pub fn mses<F: Float>(xs: &Matrix<F>, y: &Matrix<F>) -> Matrix<F> {
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    let data = xs
        .row_iter()
        .map(|x| {
            x.iter()
                .zip(y.iter())
                .fold(F::zero(), |acc, (&xi, &yi)| acc + (xi - yi).powi(2))
        })
        .map(|x| x.div(F::from(xs.cols()).unwrap()))
        .collect::<Vec<_>>();

    Matrix::from_vec1d(1, xs.rows(), data)
}

/// xs: batch_size x output_size
/// y : 1 x output_size
/// Returns a batch_size x output_size Matrix of d(MSE)/dx: 2 * (x - y) / output_size
pub fn mse_backward<F: Float>(mut xs: Matrix<F>, y: &Matrix<F>) -> Matrix<F> {
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    let two = F::from(2).unwrap();
    let output_size = F::from(xs.cols()).unwrap();

    xs.row_iter_mut().for_each(|x| {
        x.iter_mut()
            .zip(y.iter())
            .for_each(|(xi, &yi)| *xi = two * (*xi - yi) / output_size)
    });
    xs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let eps = 1e-12;
        assert!(
            (actual - expected).abs() < eps,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn mses_returns_per_sample_mean_squared_error() {
        let xs = Matrix::from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
        let y = Matrix::from_vec2d(vec![vec![1.0_f64, 2.0, 3.0]]);

        let result = mses(&xs, &y);

        assert_eq!(result.rows(), 1);
        assert_eq!(result.cols(), 2);
        assert_close(result.get(0, 0), 0.0);
        assert_close(result.get(0, 1), 1.0);
    }

    #[test]
    fn mse_backward_matches_mean_squared_error_gradient() {
        let xs = Matrix::from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![2.0, 4.0, 6.0]]);
        let y = Matrix::from_vec2d(vec![vec![1.0_f64, 1.0, 1.0]]);

        let grad = mse_backward(xs, &y);

        assert_eq!(grad.rows(), 2);
        assert_eq!(grad.cols(), 3);
        assert_close(grad.get(0, 0), 0.0);
        assert_close(grad.get(0, 1), 2.0 / 3.0);
        assert_close(grad.get(0, 2), 4.0 / 3.0);
        assert_close(grad.get(1, 0), 2.0 / 3.0);
        assert_close(grad.get(1, 1), 2.0);
        assert_close(grad.get(1, 2), 10.0 / 3.0);
    }

    #[test]
    fn mse_backward_is_elementwise_against_target_row() {
        let xs = Matrix::from_vec2d(vec![vec![2.0_f64, 4.0, 6.0]]);
        let y = Matrix::from_vec2d(vec![vec![1.0_f64, 2.0, 3.0]]);

        let grad = mse_backward(xs, &y);

        assert_close(grad.get(0, 0), 2.0 / 3.0);
        assert_close(grad.get(0, 1), 4.0 / 3.0);
        assert_close(grad.get(0, 2), 2.0);
    }
}
