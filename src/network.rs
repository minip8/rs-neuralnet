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
        let cost_backward = self.cost.backward()(output, y);
        self.backward(cost_backward);
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
            vec![Layer::<f64>::relu(3, 4), Layer::<f64>::relu(4, 2)],
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
            vec![Layer::<f64>::relu(2, 3), Layer::<f64>::relu(3, 1)],
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

    #[test]
    fn test_network_learns_xor_behavior_cross_entropy() {
        let train_samples = [
            (
                Matrix::from_vec2d(vec![vec![-1.0, -1.0]]),
                Matrix::from_vec2d(vec![vec![1.0, 0.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![-1.0, 1.0]]),
                Matrix::from_vec2d(vec![vec![0.0, 1.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![1.0, -1.0]]),
                Matrix::from_vec2d(vec![vec![0.0, 1.0]]),
            ),
            (
                Matrix::from_vec2d(vec![vec![1.0, 1.0]]),
                Matrix::from_vec2d(vec![vec![1.0, 0.0]]),
            ),
        ];

        let mut learned_xor = false;

        // Retries reduce flakiness from unlucky random initializations.
        for _ in 0..1 {
            let mut network = Network::new(
                vec![
                    Layer::<f64>::relu(2, 16),
                    Layer::<f64>::relu(16, 8),
                    Layer::<f64>::linear(8, 2),
                ],
                Cost::cross_entropy(),
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
                .max();
            let p01 = network
                .forward(Matrix::from_vec2d(vec![vec![-1.0, 1.0]]))
                .max();
            let p10 = network
                .forward(Matrix::from_vec2d(vec![vec![1.0, -1.0]]))
                .max();
            let p11 = network
                .forward(Matrix::from_vec2d(vec![vec![1.0, 1.0]]))
                .max();

            println!("{p00:?} {p01:?} {p10:?} {p11:?}");

            if p00.0 == 0 && p01.0 == 1 && p10.0 == 1 && p11.0 == 0 {
                learned_xor = true;
                break;
            }
        }

        assert!(learned_xor, "network failed to learn XOR behavior");
    }

    #[test]
    #[ignore]
    fn test_network_trains_on_mnist_digits() {
        use mnist::MnistBuilder;

        // Try to load MNIST data; skip if files are not available
        let mnist_data_available = std::fs::metadata("data/train-images-idx3-ubyte").is_ok()
            && std::fs::metadata("data/train-labels-idx1-ubyte").is_ok();

        if !mnist_data_available {
            println!("MNIST data files not found. Skipping test.");
            println!("Download MNIST data from http://yann.lecun.com/exdb/mnist/ to enable this test");
            return;
        }

        let mnist = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(2000)
            .validation_set_length(0)
            .test_set_length(500)
            .finalize();

        let train_images = mnist.trn_img;
        let train_labels = mnist.trn_lbl;
        let test_images = mnist.tst_img;
        let test_labels = mnist.tst_lbl;

        println!("Loaded {} MNIST training samples", train_labels.len());
        println!("Loaded {} MNIST test samples", test_labels.len());

        // Normalize training images to [0, 1]
        let normalized_train_images: Vec<f64> = train_images.iter().map(|&x| x as f64 / 255.0).collect();
        let normalized_test_images: Vec<f64> = test_images.iter().map(|&x| x as f64 / 255.0).collect();

        // Create training samples with one-hot encoded labels
        let mut training_samples = Vec::new();
        for i in 0..train_labels.len() {
            let start_idx = i * 784;
            let end_idx = start_idx + 784;
            let image_vec: Vec<f64> = normalized_train_images[start_idx..end_idx].to_vec();

            // One-hot encode the label
            let mut label_vec = vec![0.0; 10];
            label_vec[train_labels[i] as usize] = 1.0;

            let x = Matrix::from_vec1d(1, 784, image_vec);
            let y = Matrix::from_vec1d(1, 10, label_vec);

            training_samples.push((x, y));
        }

        // Create test samples
        let mut test_samples = Vec::new();
        for i in 0..test_labels.len() {
            let start_idx = i * 784;
            let end_idx = start_idx + 784;
            let image_vec: Vec<f64> = normalized_test_images[start_idx..end_idx].to_vec();

            let x = Matrix::from_vec1d(1, 784, image_vec);
            test_samples.push((x, test_labels[i]));
        }

        // Create network: 784 -> 128 -> 64 -> 10
        let mut network = Network::new(
            vec![
                Layer::<f64>::relu(784, 128),
                Layer::<f64>::relu(128, 64),
                Layer::<f64>::linear(64, 10),
            ],
            Cost::cross_entropy(),
            0.01,
        );

        println!("Network created with architecture: 784 -> 128 -> 64 -> 10");

        // Train for a few epochs
        for epoch in 0..10 {
            let mut epoch_loss_sum = 0.0;

            for (x, y) in &training_samples {
                let loss = network.step(x.clone(), y);
                epoch_loss_sum += loss;
            }

            let epoch_mean_loss = epoch_loss_sum / training_samples.len() as f64;
            println!("MNIST epoch {}: mean_loss={:.6}", epoch, epoch_mean_loss);
        }

        // Evaluate on test set
        let mut correct_predictions = 0;
        for (x, true_label) in &test_samples {
            let output = network.forward(x.clone());
            let (predicted_label, _) = output.max();
            if predicted_label == *true_label as usize {
                correct_predictions += 1;
            }
        }

        let test_accuracy = correct_predictions as f64 / test_samples.len() as f64;
        println!(
            "Test accuracy: {}/{} = {:.4}",
            correct_predictions,
            test_samples.len(),
            test_accuracy
        );

        println!("MNIST training and evaluation completed successfully");
    }
}
