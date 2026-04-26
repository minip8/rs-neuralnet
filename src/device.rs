pub trait Device: Clone + 'static {
    type Matrix<T>;
}
