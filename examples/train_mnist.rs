use mnist::MnistBuilder;
use rs_neuralnet::{cost::Cost, layer::Layer, matrix::{Matrix, cpu::Cpu}, network::Network};

fn main() {
    let required_files = [
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    ];

    let all_data_files_available = required_files
        .iter()
        .all(|path| std::fs::metadata(path).is_ok());

    if !all_data_files_available {
        println!("MNIST data files not found in ./data.");
        println!("Expected files:");
        for file in required_files {
            println!("- {file}");
        }
        println!("Download MNIST data to run this example.");
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

    let normalized_train_images: Vec<f64> =
        train_images.iter().map(|&x| x as f64 / 255.0).collect();
    let normalized_test_images: Vec<f64> = test_images.iter().map(|&x| x as f64 / 255.0).collect();

    let mut training_samples = Vec::new();
    for i in 0..train_labels.len() {
        let start_idx = i * 784;
        let end_idx = start_idx + 784;
        let image_vec: Vec<f64> = normalized_train_images[start_idx..end_idx].to_vec();

        let mut label_vec = vec![0.0; 10];
        label_vec[train_labels[i] as usize] = 1.0;

        let x = Matrix::from_vec1d(1, 784, image_vec);
        let y = Matrix::from_vec1d(1, 10, label_vec);

        training_samples.push((x, y));
    }

    let mut test_samples = Vec::new();
    for i in 0..test_labels.len() {
        let start_idx = i * 784;
        let end_idx = start_idx + 784;
        let image_vec: Vec<f64> = normalized_test_images[start_idx..end_idx].to_vec();

        let x = Matrix::from_vec1d(1, 784, image_vec);
        test_samples.push((x, test_labels[i]));
    }

    let mut network = Network::new(
        vec![
            Layer::<f64, Cpu>::relu(784, 128),
            Layer::<f64, Cpu>::relu(128, 64),
            Layer::<f64, Cpu>::linear(64, 10),
        ],
        Cost::cross_entropy(),
        0.01,
    );

    println!("Network created with architecture: 784 -> 128 -> 64 -> 10");

    for epoch in 0..10 {
        let mut epoch_loss_sum = 0.0;

        for (x, y) in &training_samples {
            let loss = network.step(x.clone(), y);
            epoch_loss_sum += loss;
        }

        let epoch_mean_loss = epoch_loss_sum / training_samples.len() as f64;
        println!("MNIST epoch {}: mean_loss={:.6}", epoch, epoch_mean_loss);
    }

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
