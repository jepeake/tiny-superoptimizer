use crate::tensor::Tensor;
use crate::op::{Op, OpKind};

#[derive(Debug, Clone)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    pub outputs: Vec<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self { tensors: Vec::new(), ops: Vec::new(), outputs: Vec::new() }
    }

    pub fn add_tensor(&mut self, shape: Vec<usize>, name: String) -> usize {
        self.tensors.push(Tensor { shape, name });
        self.tensors.len() - 1
    }

    pub fn add_op(&mut self, kind: OpKind, inputs: Vec<usize>, attrs: Vec<i64>) -> usize {
        let id = self.tensors.len();
        let output_shape = match kind {
            OpKind::Add | OpKind::Mul | OpKind::Mod | OpKind::LessThan => self.tensors[inputs[0]].shape.clone(),
            OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Sqrt | OpKind::Recip => self.tensors[inputs[0]].shape.clone(),
            OpKind::SumReduce | OpKind::MaxReduce => {
                let mut shape = self.tensors[inputs[0]].shape.clone();
                let axis = attrs.get(0).map(|&x| x as usize).filter(|&x| x < shape.len());
                axis.map(|a| shape.remove(a)).or_else(|| shape.pop());
                shape
            },
            OpKind::Reshape => attrs.iter().map(|&x| x as usize).collect(),
            OpKind::Broadcast => if attrs.is_empty() { self.tensors[inputs[0]].shape.clone() } else { attrs.iter().map(|&x| x as usize).collect() },
            OpKind::Constant | OpKind::Input(_) => vec![],
        };
        self.tensors.push(Tensor {
            shape: output_shape,
            name: format!("tmp_{}", id),
        });
        self.ops.push(Op { kind, inputs, attrs });
        id
    }

    pub fn format_program(&self) -> String {
        let mut lines = Vec::new();
        
        // Inputs
        for tensor in self.tensors.iter() {
            if !tensor.name.starts_with("tmp_") {
                let shape_str = if tensor.shape.is_empty() { "scalar".to_string() } else { format!("[{}]", tensor.shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")) };
                lines.push(format!("{}: {} = input", tensor.name, shape_str));
            }
        }
        
        // Ops
        for (op_idx, op) in self.ops.iter().enumerate() {
            let output_tensor_idx = self.tensors.len() - self.ops.len() + op_idx;
            let output_name = &self.tensors[output_tensor_idx].name;
            let output_shape = &self.tensors[output_tensor_idx].shape;
            
            let shape_str = if output_shape.is_empty() { "scalar".to_string() } else { format!("[{}]", output_shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")) };
            
            let input_names: Vec<String> = op.inputs.iter()
                .map(|&idx| self.tensors[idx].name.clone())
                .collect();
            
            let op_str = match &op.kind {
                OpKind::Add => format!("{} + {}", input_names[0], input_names[1]),
                OpKind::Mul => format!("{} * {}", input_names[0], input_names[1]),
                OpKind::SumReduce => op.attrs.get(0).map_or(format!("sum({})", input_names[0]), |&a| format!("sum({}, axis={})", input_names[0], a)),
                OpKind::MaxReduce => op.attrs.get(0).map_or(format!("max({})", input_names[0]), |&a| format!("max({}, axis={})", input_names[0], a)),
                OpKind::Reshape => format!("reshape({}, [{}])", input_names[0], op.attrs.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")),
                OpKind::Broadcast => if op.attrs.is_empty() { format!("broadcast({})", input_names[0]) } else { format!("broadcast({}, [{}])", input_names[0], op.attrs.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")) },
                OpKind::Exp => format!("exp({})", input_names[0]),
                OpKind::Log => format!("log({})", input_names[0]),
                OpKind::Sin => format!("sin({})", input_names[0]),
                OpKind::Sqrt => format!("sqrt({})", input_names[0]),
                OpKind::Recip => format!("1 / {}", input_names[0]),
                OpKind::Mod => format!("{} % {}", input_names[0], input_names[1]),
                OpKind::LessThan => format!("{} < {}", input_names[0], input_names[1]),
                OpKind::Constant => format!("constant({})", op.attrs.get(0).unwrap_or(&0)),
                OpKind::Input(name) => format!("input({})", name),
            };
            
            lines.push(format!("{}: {} = {}", output_name, shape_str, op_str));
        }
        
        if !self.outputs.is_empty() {
            let output_names: Vec<String> = self.outputs.iter()
                .map(|&idx| self.tensors[idx].name.clone())
                .collect();
            lines.push(format!("return {}", output_names.join(", ")));
        }
        
        lines.join("\n")
    }
} 