use num_traits::Float;

use crate::device::Device;

#[derive(Clone)]
pub struct Cpu;

pub struct CpuBuffer<F> {
    data: Vec<F>,
}

impl Device for Cpu {
    type Buffer<F: Float> = CpuBuffer<F>;
}
