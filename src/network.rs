use num_traits::Float;

use crate::{layer::Layer, matrix::Matrix};

pub struct Network<T> {
    layers: Vec<Layer<T>>,
}

impl<F> Network<F> where F: Float {

}
