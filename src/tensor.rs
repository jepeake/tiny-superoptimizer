
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub name: String,
}

pub fn tensor_size(shape: &[usize]) -> usize {
    shape.iter().product()
} 