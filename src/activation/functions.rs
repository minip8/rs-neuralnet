use num_traits::Float;

use crate::matrix::{Matrix, cpu::Cpu};

pub fn relu<F: Float>(x: F) -> F {
    x.max(F::zero())
}

pub fn relu_backward<F: Float>(x: F) -> F {
    if x > F::zero() { F::one() } else { F::zero() }
}

pub fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

pub fn sigmoid_backward<F: Float>(x: F) -> F {
    let s = sigmoid(x);
    s * (F::one() - s)
}

pub fn tanh<F: Float>(x: F) -> F {
    x.tanh()
}

pub fn tanh_backward<F: Float>(x: F) -> F {
    F::one() - x.tanh().powi(2)
}

pub fn linear<F: Float>(x: F) -> F {
    x
}

pub fn linear_backward<F: Float>(_: F) -> F {
    F::one()
}

/// Applies softmax to every row independently
pub fn softmax<F: Float>(mut m: Matrix<F, Cpu>) -> Matrix<F, Cpu> {
    m.row_iter_mut().for_each(|r| {
        let mx = *r.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let s = r.iter().fold(F::zero(), |acc, &x| acc + (x - mx).exp());
        r.iter_mut().for_each(|x| *x = (*x - mx).exp() / s);
    });
    m
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
    fn softmax_rows_sum_to_one_and_are_finite() {
        let xs = Matrix::from_vec2d(vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![1000.0, 1001.0, 1002.0],
            vec![-1000.0, -1001.0, -1002.0],
        ]);

        let probs = softmax(xs);

        for row in probs.row_iter() {
            let sum = row.iter().copied().fold(0.0, |acc, x| acc + x);
            assert_close(sum, 1.0);

            for &p in row.iter() {
                assert!(p.is_finite());
                assert!(p >= 0.0);
                assert!(p <= 1.0);
            }
        }
    }

    #[test]
    fn softmax_is_shift_invariant_per_row() {
        let base = Matrix::from_vec2d(vec![vec![1.5_f64, -0.5, 3.0], vec![2.0, 4.0, -1.0]]);
        let shifted = Matrix::from_vec2d(vec![
            vec![1.5_f64 + 10000.0, -0.5 + 10000.0, 3.0 + 10000.0],
            vec![2.0 - 7777.0, 4.0 - 7777.0, -1.0 - 7777.0],
        ]);

        let base_probs = softmax(base);
        let shifted_probs = softmax(shifted);

        for i in 0..base_probs.rows() {
            for j in 0..base_probs.cols() {
                assert_close(base_probs.get(i, j), shifted_probs.get(i, j));
            }
        }
    }
}
