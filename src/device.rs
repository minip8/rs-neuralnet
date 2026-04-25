use num_traits::Float;

pub mod cpu;
pub mod cuda;

pub trait Device: Clone + 'static {
    type Buffer<F: Float>;
}
