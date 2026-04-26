use std::marker::PhantomData;

#[derive(Clone)]
pub struct RowIter<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    next_row: usize,
}

pub struct RowIterMut<'a, T> {
    inner: RowIterMutInner<'a, T>,
}

enum RowIterMutInner<'a, T> {
    Chunks(std::slice::ChunksExactMut<'a, T>),
    ZeroCols {
        ptr: *mut T,
        remaining_rows: usize,
        _marker: PhantomData<&'a mut T>,
    },
}

impl<'a, T> RowIter<'a, T> {
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self {
            data,
            cols,
            rows,
            next_row: 0,
        }
    }
}

impl<'a, T> RowIterMut<'a, T> {
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        if cols == 0 {
            let ptr = data.as_mut_ptr();
            return Self {
                inner: RowIterMutInner::ZeroCols {
                    ptr,
                    remaining_rows: rows,
                    _marker: PhantomData,
                },
            };
        }

        Self {
            inner: RowIterMutInner::Chunks(data.chunks_exact_mut(cols)),
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

impl<'a, T> Iterator for RowIterMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RowIterMutInner::Chunks(iter) => iter.next(),
            RowIterMutInner::ZeroCols {
                ptr,
                remaining_rows,
                ..
            } => {
                if *remaining_rows == 0 {
                    return None;
                }

                *remaining_rows -= 1;
                // Safe because the slice is always empty, so it never aliases data.
                Some(unsafe { std::slice::from_raw_parts_mut(*ptr, 0) })
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            RowIterMutInner::Chunks(iter) => iter.size_hint(),
            RowIterMutInner::ZeroCols { remaining_rows, .. } => {
                (*remaining_rows, Some(*remaining_rows))
            }
        }
    }
}

impl<'a, T> ExactSizeIterator for RowIterMut<'a, T> {}

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

    #[test]
    fn row_iter_mut_yields_mutable_rows() {
        let mut m = Matrix::from_vec2d(vec![vec![1, 2], vec![3, 4]]);

        for row in m.row_iter_mut() {
            row[0] *= 10;
        }

        assert_eq!(m.data(), &vec![10, 2, 30, 4]);
    }
}
