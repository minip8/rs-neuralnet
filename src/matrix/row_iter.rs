use super::Matrix;

#[derive(Clone)]
pub struct RowIter<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    next_row: usize,
}

impl<'a, T> RowIter<'a, T> {
    pub fn new(matrix: &'a Matrix<T>) -> Self {
        Self {
            data: matrix.data(),
            cols: matrix.cols(),
            next_row: 0,
            rows: matrix.rows(),
        }
    }
}

impl<'a, T> Iterator for RowIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.rows {
            return None;
        }

        // For zero-column matrices, yield one empty slice per row.
        if self.cols == 0 {
            self.next_row += 1;
            return Some(&self.data[0..0]);
        }

        let start = self.next_row * self.cols;
        let end = start + self.cols;
        self.next_row += 1;

        Some(&self.data[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rows.saturating_sub(self.next_row);
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for RowIter<'a, T> {}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn row_iter_yields_each_row_slice() {
        let m = Matrix::from_vec2d(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let mut iter = m.row_iter();

        assert_eq!(iter.next(), Some(&[1, 2, 3][..]));
        assert_eq!(iter.next(), Some(&[4, 5, 6][..]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn row_iter_handles_zero_cols() {
        let m: Matrix<f32> = Matrix::zeros(3, 0);
        let rows: Vec<&[f32]> = m.row_iter().collect();

        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.is_empty()));
    }

    #[test]
    fn row_iter_can_cycle() {
        let m = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);
        let rows: Vec<&[i32]> = m.row_iter().cycle().take(5).collect();

        assert_eq!(rows[0], &[1, 2]);
        assert_eq!(rows[1], &[3, 4]);
        assert_eq!(rows[2], &[1, 2]);
        assert_eq!(rows[3], &[3, 4]);
        assert_eq!(rows[4], &[1, 2]);
    }
}
