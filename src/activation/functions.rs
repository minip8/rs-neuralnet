use num_traits::Float;

pub fn relu<T: Float>(x: T) -> T {
    x.max(T::zero())
}

pub fn relu_backward<T: Float>(x: T) -> T {
    if x > T::zero() { T::one() } else { T::zero() }
}

pub fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

pub fn sigmoid_backward<T: Float>(x: T) -> T {
    let s = sigmoid(x);
    s * (T::one() - s)
}

pub fn tanh<T: Float>(x: T) -> T {
    x.tanh()
}

pub fn tanh_backward<T: Float>(x: T) -> T {
    T::one() - x.tanh().powi(2)
}
