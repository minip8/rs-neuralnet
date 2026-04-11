use num_traits::Float;

pub mod functions;

pub struct Activation<T, F1, F2> {
    forward: F1,
    backward: F2,
    _marker: std::marker::PhantomData<T>,
}

impl<T, F1, F2> Activation<T, F1, F2>
where
    F1: Fn(T) -> T,
    F2: Fn(T) -> T,
{
    pub fn new(forward: F1, backward: F2) -> Activation<T, F1, F2> {
        Activation {
            forward,
            backward,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Activation<T, fn(T) -> T, fn(T) -> T>
where
    T: Float,
{
    pub fn relu() -> Self {
        Activation::new(functions::relu, functions::relu_backward)
    }

    pub fn sigmoid() -> Self {
        Activation::new(functions::sigmoid, functions::sigmoid_backward)
    }

    pub fn tanh() -> Self {
        Activation::new(functions::tanh, functions::tanh_backward)
    }

    pub fn linear() -> Self {
        Activation::new(functions::linear, functions::linear_backward)
    }

    pub fn forward(&self, x: T) -> T {
        (self.forward)(x)
    }

    pub fn backward(&self, x: T) -> T {
        (self.backward)(x)
    }
}
