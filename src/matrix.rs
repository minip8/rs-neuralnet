pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0f32; rows * cols],
        }
    }

    pub fn from_vec1d(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        Matrix::assert_rowcol_dimensions_match_data(rows, cols, &data);

        Matrix { rows, cols, data }
    }

    pub fn from_vec2d(data2d: Vec<Vec<f32>>) -> Matrix {
        debug_assert!(data2d.len() > 0);

        let rows = data2d.len();
        let cols = data2d[0].len();
        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.set_(i, j, data2d[i][j]);
            }
        }
        res
    }
}

// Assertions
impl Matrix {
    fn assert_rowcol_dimensions_match_data(rows: usize, cols: usize, data: &Vec<f32>) {
        debug_assert_eq!(rows * cols, data.len())
    }

    fn assert_same_dimensions(&self, other: &Matrix) {
        debug_assert_eq!((self.rows, self.cols), (other.rows, other.cols));
    }

    fn assert_matmul_compatible(&self, other: &Matrix) {
        debug_assert_eq!(self.cols, other.rows);
    }

    fn assert_idx_ok(&self, idx: usize) {
        debug_assert!(idx < self.data.len());
    }
}

// QOL
impl Matrix {
    fn rowcol_to_idx(&self, r: usize, c: usize) -> usize {
        let idx = r * self.cols + c;
        self.assert_idx_ok(idx);
        idx
    }

    fn get(&self, r: usize, c: usize) -> f32 {
        self.data[self.rowcol_to_idx(r, c)]
    }

    fn set_(&mut self, r: usize, c: usize, v: f32) {
        let idx = self.rowcol_to_idx(r, c);
        self.data[idx] = v;
    }
}

// Unary operations
impl Matrix {
    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set_(j, i, self.get(i, j));
            }
        }
        res
    }
}

// Binary operations
impl Matrix {
    pub fn add_(&mut self, other: &Matrix) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a += b);
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }

    pub fn sub_(&mut self, other: &Matrix) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a -= b);
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }

    pub fn dot_(&mut self, other: &Matrix) {
        self.assert_same_dimensions(other);

        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(a, b)| *a *= b);
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        self.assert_same_dimensions(other);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }

    pub fn mul_(&mut self, scale: f32) {
        self.data.iter_mut().for_each(|a| *a *= scale);
    }

    pub fn mul(&self, scale: f32) -> Matrix {
        let data = self.data.iter().map(|a| a * scale).collect::<Vec<_>>();

        Matrix::from_vec1d(self.rows, self.cols, data)
    }

    pub fn mat_mul_(&mut self, other: &Matrix) {
        self.assert_matmul_compatible(other);

        let mut data = vec![0f32; self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.rows + j] += self.get(i, k) * other.get(k, j);
                }
            }
        }
        self.cols = other.cols;
        self.data = data;
    }

    pub fn mat_mul(&self, other: &Matrix) -> Matrix {
        self.assert_matmul_compatible(other);

        let mut data = vec![0f32; self.rows * other.cols];

        for i in 0..self.rows {
            for k in 0..other.rows {
                for j in 0..other.cols {
                    data[i * other.rows + j] += self.get(i, k) * other.get(k, j);
                }
            }
        }
        Matrix::from_vec1d(self.rows, other.cols, data)
    }
}
