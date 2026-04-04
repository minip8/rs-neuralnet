use num_traits::Float;

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
