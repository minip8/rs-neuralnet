use num_traits::Float;

use crate::{activation::functions::softmax, matrix::{Matrix, cpu::Cpu}};

fn apply_elementwise<T, F>(mut xs: Matrix<T, Cpu>, y: &Matrix<T, Cpu>, f: F) -> Matrix<T, Cpu>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    xs.row_iter_mut().for_each(|x| {
        x.iter_mut()
            .zip(y.iter())
            .for_each(|(xi, &yi)| *xi = f(*xi, yi))
    });
    xs
}

/// xs: batch_size x output_size
/// y : 1 x output_size
/// Returns the average of the MSEs of each individual batch
pub fn mse<F: Float>(xs: &Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> F {
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    xs.row_iter()
        .map(|x| {
            x.iter()
                .zip(y.iter())
                .fold(F::zero(), |acc, (&xi, &yi)| acc + (xi - yi).powi(2))
        })
        .fold(F::zero(), |acc, x| acc + x)
        .div(F::from(xs.cols()).unwrap())
        .div(F::from(xs.rows()).unwrap())
}

/// xs: batch_size x output_size
/// y : 1 x output_size
/// Returns a 1 x batch_size row vector of the MSEs of all individual batches
pub fn mses<F: Float>(xs: &Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> Matrix<F, Cpu> {
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
/// Returns a batch_size x output_size Matrix MSE: (x - y)^2 / output_size
pub fn mse_forward<F: Float>(xs: Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> Matrix<F, Cpu> {
    let output_size = F::from(xs.cols()).unwrap();

    apply_elementwise(xs, y, |xi, yi| (xi - yi).powi(2) / output_size)
}

/// xs: batch_size x output_size
/// y : 1 x output_size
/// Returns a batch_size x output_size Matrix of d(MSE)/dx: 2 * (x - y) / output_size
pub fn mse_backward<F: Float>(xs: Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> Matrix<F, Cpu> {
    let two = F::from(2).unwrap();
    let output_size = F::from(xs.cols()).unwrap();

    apply_elementwise(xs, y, |xi, yi| two * (xi - yi) / output_size)
}

/// xs: batch_size x output_size
/// y: 1 x output_size (1-hot vector)
/// Returns a 1 x batch_size row vector
pub fn cross_entropy_loss_forward<F: Float>(xs: &Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> Matrix<F, Cpu> {
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    let y = y.max().0;

    let data = xs
        .row_iter()
        .map(|r| {
            let mx = *r.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let exp_sum_shift = r.iter().fold(F::zero(), |acc, &x| acc + (x - mx).exp());
            let log_sum = mx + exp_sum_shift.ln();

            log_sum - r[y]
        })
        .collect::<Vec<_>>();

    Matrix::from_vec1d(1, xs.rows(), data)
}

/// xs: batch_size x output_size
/// y: 1 x output_size (1-hot vector)
/// Applies softmax and then computes d(cross entropy loss)/dz for each batch
/// Returns a batch_size x output_size row vector
pub fn cross_entropy_loss_backward<F: Float>(xs: Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> Matrix<F, Cpu> {
    debug_assert_eq!(y.rows(), 1);
    debug_assert_eq!(xs.cols(), y.cols());

    let mut xs = softmax(xs);
    let y = y.max().0;

    xs.row_iter_mut().for_each(|r| r[y] = r[y] - F::one());
    xs
}

/// Returns the mean cross entropy loss across all batches
pub fn cross_entropy_loss<F: Float>(xs: &Matrix<F, Cpu>, y: &Matrix<F, Cpu>) -> F {
    cross_entropy_loss_forward(xs, y)
        .iter()
        .fold(F::zero(), |acc, &x| acc + x)
        .div(F::from(xs.rows()).unwrap())
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

    #[test]
    fn apply_elementwise_applies_target_row_to_each_batch_row() {
        let xs = Matrix::from_vec2d(vec![vec![1_i32, 2, 3], vec![10, 20, 30]]);
        let y = Matrix::from_vec2d(vec![vec![1_i32, 10, 100]]);

        let result = apply_elementwise(xs, &y, |xi, yi| xi + yi);

        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 3);
        assert_eq!(result.get(0, 0), 2);
        assert_eq!(result.get(0, 1), 12);
        assert_eq!(result.get(0, 2), 103);
        assert_eq!(result.get(1, 0), 11);
        assert_eq!(result.get(1, 1), 30);
        assert_eq!(result.get(1, 2), 130);
    }

    #[test]
    fn apply_elementwise_supports_non_commutative_operation() {
        let xs = Matrix::from_vec2d(vec![vec![5_i32, 7, 11], vec![13, 17, 19]]);
        let y = Matrix::from_vec2d(vec![vec![1_i32, 2, 3]]);

        let result = apply_elementwise(xs, &y, |xi, yi| xi - 2 * yi);

        assert_eq!(result.get(0, 0), 3);
        assert_eq!(result.get(0, 1), 3);
        assert_eq!(result.get(0, 2), 5);
        assert_eq!(result.get(1, 0), 11);
        assert_eq!(result.get(1, 1), 13);
        assert_eq!(result.get(1, 2), 13);
    }

    #[test]
    fn cross_entropy_forward_matches_known_logsumexp_form() {
        let xs = Matrix::from_vec2d(vec![vec![2.0_f64, 1.0, 0.0], vec![0.0, 0.0, 0.0]]);
        let y = Matrix::from_vec2d(vec![vec![1.0_f64, 0.0, 0.0]]);

        let losses = cross_entropy_loss_forward(&xs, &y);

        let expected_row0 = (2.0_f64.exp() + 1.0_f64.exp() + 0.0_f64.exp()).ln() - 2.0;
        let expected_row1 = (0.0_f64.exp() + 0.0_f64.exp() + 0.0_f64.exp()).ln() - 0.0;

        assert_eq!(losses.rows(), 1);
        assert_eq!(losses.cols(), 2);
        assert_close(losses.get(0, 0), expected_row0);
        assert_close(losses.get(0, 1), expected_row1);
    }

    #[test]
    fn cross_entropy_loss_is_mean_over_batch_rows() {
        let xs = Matrix::from_vec2d(vec![vec![2.0_f64, 1.0, 0.0], vec![0.0, 0.0, 0.0]]);
        let y = Matrix::from_vec2d(vec![vec![1.0_f64, 0.0, 0.0]]);

        let losses = cross_entropy_loss_forward(&xs, &y);
        let manual_mean = (losses.get(0, 0) + losses.get(0, 1)) / 2.0;

        assert_close(cross_entropy_loss(&xs, &y), manual_mean);
    }

    #[test]
    fn cross_entropy_backward_matches_softmax_minus_one_hot() {
        let xs = Matrix::from_vec2d(vec![vec![2.0_f64, 1.0, 0.0], vec![0.0, 0.0, 0.0]]);
        let y = Matrix::from_vec2d(vec![vec![0.0_f64, 1.0, 0.0]]);

        let grad = cross_entropy_loss_backward(xs.clone(), &y);
        let mut expected = softmax(xs);
        expected.row_iter_mut().for_each(|r| r[1] = r[1] - 1.0);

        assert_eq!(grad.rows(), expected.rows());
        assert_eq!(grad.cols(), expected.cols());

        for i in 0..grad.rows() {
            for j in 0..grad.cols() {
                assert_close(grad.get(i, j), expected.get(i, j));
            }
        }
    }
}
