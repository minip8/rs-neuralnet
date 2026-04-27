use num_traits::Float;

pub trait MatrixOps<T> {
    fn new(data: Vec<T>, rows: usize, cols: usize) -> Self;

    fn fill(v: T, rows: usize, cols: usize) -> Self
    where
        T: Copy;

    fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Float;

    fn numel(&self) -> usize;

    fn rows(&self) -> usize;

    fn cols(&self) -> usize;

    fn get(&self, r: usize, c: usize) -> T
    where
        T: Copy;

    fn set_(&mut self, r: usize, c: usize, v: T) -> &mut Self;

    fn set_row_(&mut self, r: usize, values: &[T]) -> &mut Self
    where
        T: Copy;

    fn apply_<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&T) -> T;

    fn transpose(&self) -> Self
    where
        T: Copy;

    fn add_(&mut self, other: &Self) -> &mut Self
    where
        T: Float;

    fn sub_(&mut self, other: &Self) -> &mut Self
    where
        T: Float;

    fn hadamard_(&mut self, other: &Self) -> &mut Self
    where
        T: Float;

    fn mul_(&mut self, scale: T) -> &mut Self
    where
        T: Float;

    fn dot(&self, other: &Self) -> T
    where
        T: Float;

    fn mat_mul(&self, other: &Self) -> Self
    where
        T: Float;

    fn add_row_to_all_rows_(&mut self, other: &Self) -> &mut Self
    where
        T: Float;

    fn col_sum_(&mut self) -> &mut Self
    where
        T: Float;

    fn max(&self) -> (usize, T)
    where
        T: Float;
}
