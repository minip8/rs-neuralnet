use num_traits::Float;

use crate::device::Device;

#[derive(Clone)]
pub struct Cuda;

pub struct CudaBuffer<F> {
    ptr: *mut F,
    len: usize,
}

impl Device for Cuda {
    type Buffer<F: Float> = CudaBuffer<F>;
}
