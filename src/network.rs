use std::fmt::Display;

use num_traits::Float;

use crate::{cost::Cost, layer::Layer, matrix::Matrix};

pub struct Network<T> {
    layers: Vec<Layer<T>>,
    cost: Cost<T>,
    lr: T,
}

impl<F> Network<F>
where
    F: Float + Display,
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
        debug_assert!(self.layers.len() >= 1);

        for i in (0..self.layers.len()).rev() {
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

#[cfg(test)]
mod tests {
    use num_traits::abs;

    use super::*;
    use crate::{cost::Cost, layer::Layer, matrix::Matrix};

    #[test]
    fn test_network_forward_output_shape() {
        let mut network = Network::new(
            vec![
                Layer::<f64>::relu(3, 4),
                Layer::<f64>::relu(4, 2),
            ],
            Cost::mse(),
            0.01,
        );

        let x = Matrix::from_vec2d(vec![vec![0.5, -1.0, 2.0], vec![1.5, 0.0, -0.5]]);
        let out = network.forward(x);

        assert_eq!(out.rows(), 2);
        assert_eq!(out.cols(), 2);
    }

    #[test]
    fn test_network_step_produces_finite_non_negative_loss() {
        let mut network = Network::new(
            vec![
                Layer::<f64>::relu(2, 3),
                Layer::<f64>::relu(3, 1),
            ],
            Cost::mse(),
            0.01,
        );

        let x = Matrix::from_vec2d(vec![vec![1.0, -1.0], vec![0.5, 2.0], vec![-1.5, 0.25]]);
        let y = Matrix::from_vec2d(vec![vec![0.0]]);

        for _ in 0..10 {
            let loss = network.step(x.clone(), &y);
            assert!(loss.is_finite());
            assert!(loss >= 0.0);
        }
    }

    #[test]
    fn test_network_can_predict_constant_value_after_training() {
        let x_train = Matrix::from_vec2d(vec![vec![0.5], vec![1.0], vec![2.0], vec![3.0]]);
        let y_train = Matrix::from_vec2d(vec![vec![1.0]]);
        let x_test = Matrix::from_vec2d(vec![vec![1.5]]);

        let mut learned_prediction_is_close = false;

        // Initialization is random; allow a handful of retries to avoid dead-ReLU starts.
        for _ in 0..6 {
            let mut network = Network::new(
                vec![
                    Layer::<f64>::relu(1, 16),
                    Layer::<f64>::relu(16, 8),
                    Layer::<f64>::linear(8, 1),
                ],
                Cost::mse(),
                0.01,
            );

            for _ in 0..2000 {
                network.step(x_train.clone(), &y_train);
            }

            let prediction = network.forward(x_test.clone()).get(0, 0);
            if (prediction - 1.0).abs() < 0.2 {
                learned_prediction_is_close = true;
                break;
            }
        }

        assert!(
            learned_prediction_is_close,
            "network failed to learn a near-constant prediction"
        );
    }

    #[test]
    fn test_network_learns_xor_behavior() {
        let train_samples = [
            (
                Matrix::from_vec2d(vec![vec![-1.0, -1.0]]),
                Matrix::from_vec2d(vec![vec![0.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![-1.0, 1.0]]),
                Matrix::from_vec2d(vec![vec![1.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![1.0, -1.0]]),
                Matrix::from_vec2d(vec![vec![1.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![1.0, 1.0]]),
                Matrix::from_vec2d(vec![vec![0.0]]),
            ),
        ];

        let mut learned_xor = false;

        // Retries reduce flakiness from unlucky random initializations.
        for _ in 0..1 {
            let mut network = Network::new(
                vec![
                    Layer::<f64>::relu(2, 16),
                    Layer::<f64>::relu(16, 8),
                    Layer::<f64>::linear(8, 1),
                ],
                Cost::mse(),
                0.01,
            );

            for e in 0..1000 {
                let mut epoch_loss_sum = 0f64;

                for (x, y) in train_samples.iter() {
                    let loss = network.step(x.clone(), y);
                    epoch_loss_sum += loss;
                }

                if e % 500 == 0 {
                    let epoch_mean_loss = epoch_loss_sum / train_samples.len() as f64;
                    println!("epoch {e}: mean_loss={epoch_mean_loss}");
                }
            }

            let p00 = network
                .forward(Matrix::from_vec2d(vec![vec![-1.0, -1.0]]))
                .get(0, 0);
            let p01 = network
                .forward(Matrix::from_vec2d(vec![vec![-1.0, 1.0]]))
                .get(0, 0);
            let p10 = network
                .forward(Matrix::from_vec2d(vec![vec![1.0, -1.0]]))
                .get(0, 0);
            let p11 = network
                .forward(Matrix::from_vec2d(vec![vec![1.0, 1.0]]))
                .get(0, 0);

            println!("{p00:?} {p01:?} {p10:?} {p11:?}");

            let eps = 0.01;
            if (abs(p00 - 0.0) < eps)
                && (abs(p01 - 1.0) < eps)
                && (abs(p10 - 1.0) < eps)
                && (abs(p11 - 0.0) < eps)
            {
                learned_xor = true;
                break;
            }
        }

        assert!(learned_xor, "network failed to learn XOR behavior");
    }
}
