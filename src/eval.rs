use ndarray::{ArrayD, Axis};
use std::collections::HashMap;
use rand::Rng;
use crate::graph::Graph;
use crate::op::{Op, OpKind};
use crate::tensor::tensor_size;

fn broadcast_arrays<T: Clone + Default>(a: &ArrayD<T>, b: &ArrayD<T>) -> (ArrayD<T>, ArrayD<T>) {
    if a.shape() == b.shape() { return (a.clone(), b.clone()); }
    
    if a.len() == 1 && b.len() > 1 {
        return (ArrayD::from_elem(b.shape(), a.iter().next().unwrap().clone()), b.clone());
    }
    if b.len() == 1 && a.len() > 1 {
        return (a.clone(), ArrayD::from_elem(a.shape(), b.iter().next().unwrap().clone()));
    }
    
    if a.len() >= b.len() {
        (a.clone(), ArrayD::from_elem(a.shape(), b.iter().next().unwrap_or(&T::default()).clone()))
    } else {
        (ArrayD::from_elem(b.shape(), a.iter().next().unwrap_or(&T::default()).clone()), b.clone())
    }
}

pub fn flop_cost(op: &Op, g: &Graph) -> usize {
    match op.kind {
        OpKind::Add | OpKind::Mul | OpKind::Mod | OpKind::LessThan => tensor_size(&g.tensors[op.inputs[0]].shape),
        OpKind::SumReduce | OpKind::MaxReduce => tensor_size(&g.tensors[op.inputs[0]].shape),
        OpKind::Reshape | OpKind::Broadcast => 0,
        OpKind::Exp | OpKind::Log | OpKind::Sin => tensor_size(&g.tensors[op.inputs[0]].shape) * 10,
        OpKind::Sqrt | OpKind::Recip => tensor_size(&g.tensors[op.inputs[0]].shape) * 5,
        OpKind::Constant => 0,
        OpKind::Input(_) => 0,
    }
}

// Helper Functions

pub fn eval(graph: &Graph, inputs: &HashMap<String, ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>, Box<dyn std::error::Error>> {
    let mut values: Vec<ArrayD<f32>> = Vec::new();
    for tensor in &graph.tensors {
        if let Some(input_value) = inputs.get(&tensor.name) {
            values.push(input_value.clone());
        } else {
            values.push(ArrayD::zeros(tensor.shape.as_slice()));
        }
    }
    for op in &graph.ops {
        let out = match op.kind {
            OpKind::Add => { let (a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]); &a + &b },
            OpKind::Mul => { let (a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]); &a * &b },
            OpKind::SumReduce => values[op.inputs[0]].sum_axis(Axis(op.attrs.get(0).map_or(0, |&x| x as usize))),
            OpKind::MaxReduce => values[op.inputs[0]].fold_axis(Axis(op.attrs.get(0).map_or(0, |&x| x as usize)), f32::NEG_INFINITY, |&a, &b| a.max(b)),
            OpKind::Reshape => values[op.inputs[0]].clone().into_shape(op.attrs.iter().map(|&x| x as usize).collect::<Vec<_>>())?,
            OpKind::Broadcast => values[op.inputs[0]].clone(),
            OpKind::Sin => values[op.inputs[0]].mapv(|x| x.sin()),
            OpKind::Sqrt => values[op.inputs[0]].mapv(|x| x.sqrt()),
            OpKind::Recip => values[op.inputs[0]].mapv(|x| 1.0 / x),
            OpKind::Exp => values[op.inputs[0]].mapv(|x| x.exp()),
            OpKind::Log => values[op.inputs[0]].mapv(|x| x.ln()),
            OpKind::Mod => {
                let (mut a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]);
                a.zip_mut_with(&b, |x, &y| *x = *x % y);
                a
            },
            OpKind::LessThan => {
                let (mut a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]);
                a.zip_mut_with(&b, |x, &y| *x = if *x < y { 1.0 } else { 0.0 });
                a
            },
            OpKind::Constant => ArrayD::from_elem(vec![], 1.0),
            OpKind::Input(_) => panic!("Input nodes should not appear in evaluation"),
        };
        values.push(out);
    }
    Ok(graph.outputs.iter().map(|&i| values[i].clone()).collect())
}



fn eval_mod(graph: &Graph, inputs: &HashMap<String, ArrayD<i64>>, modulus: i64) -> Vec<ArrayD<i64>> {
    let mut values: Vec<ArrayD<i64>> = Vec::new();
    for tensor in &graph.tensors {
        if let Some(input_value) = inputs.get(&tensor.name) {
            values.push(input_value.clone());
        } else {
            values.push(ArrayD::zeros(tensor.shape.as_slice()));
        }
    }
    for op in &graph.ops {
        let out = match op.kind {
            OpKind::Add => { let (a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]); (&a + &b).mapv(|x| x % modulus) },
            OpKind::Mul => { let (a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]); (&a * &b).mapv(|x| x % modulus) },
            OpKind::SumReduce => values[op.inputs[0]].sum_axis(Axis(op.attrs.get(0).map_or(0, |&x| x as usize))).mapv(|x| x % modulus),
            OpKind::MaxReduce => values[op.inputs[0]].fold_axis(Axis(op.attrs.get(0).map_or(0, |&x| x as usize)), i64::MIN, |&a, &b| (a.max(b)) % modulus),
            OpKind::Reshape => {
                let new_shape: Vec<usize> = op.attrs.iter().map(|&x| x as usize).collect();
                let original = &values[op.inputs[0]];
                let same_size = original.shape().iter().product::<usize>() == new_shape.iter().product::<usize>();
                if same_size { original.clone().into_shape(new_shape).unwrap_or_else(|_| original.clone()) } else { original.clone() }
            },
            OpKind::Broadcast => values[op.inputs[0]].clone(),
            OpKind::Sin => values[op.inputs[0]].mapv(|x| (x * x) % modulus),
            OpKind::Sqrt => values[op.inputs[0]].mapv(|x| x % modulus),
            OpKind::Recip => values[op.inputs[0]].mapv(|x| if x != 0 { 1 } else { 0 }),
            OpKind::Exp => values[op.inputs[0]].mapv(|x| (x * x) % modulus),
            OpKind::Log => values[op.inputs[0]].mapv(|x| x % modulus),
            OpKind::Mod => {
                let (mut a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]);
                a.zip_mut_with(&b, |x, &y| *x = if y != 0 { (*x % y) % modulus } else { 0 });
                a
            },
            OpKind::LessThan => {
                let (mut a, b) = broadcast_arrays(&values[op.inputs[0]], &values[op.inputs[1]]);
                a.zip_mut_with(&b, |x, &y| *x = if *x < y { 1 } else { 0 });
                a
            },
            OpKind::Constant => ArrayD::from_elem(vec![], 1i64),
            OpKind::Input(_) => panic!("Input nodes should not appear in evaluation"),
        };
        values.push(out);
    }
    graph.outputs.iter().map(|&i| values[i].clone()).collect()
}

fn random_bindings(graph: &Graph, rng: &mut impl Rng, modulus: i64) -> HashMap<String, ArrayD<i64>> {
    let mut bindings = HashMap::new();
    for tensor in &graph.tensors {
        if !tensor.name.starts_with("tmp_") {
            let mut arr = ArrayD::zeros(tensor.shape.as_slice());
            arr.mapv_inplace(|_| rng.gen_range(0..modulus));
            bindings.insert(tensor.name.clone(), arr);
        }
    }
    bindings
}

pub fn equivalent(g1: &Graph, g2: &Graph, trials: usize) -> bool {
    let prime: i64 = 12_289;
    let mut rng = rand::thread_rng();
    for _ in 0..trials {
        let inputs = random_bindings(g1, &mut rng, prime);
        if let (Ok(result1), Ok(result2)) = (eval_mod_safe(g1, &inputs, prime), eval_mod_safe(g2, &inputs, prime)) {
            if result1 != result2 {
                return false;
            }
        }
    }
    true
}

fn eval_mod_safe(graph: &Graph, inputs: &HashMap<String, ArrayD<i64>>, modulus: i64) -> Result<Vec<ArrayD<i64>>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_mod(graph, inputs, modulus))).map_err(|_| "Evaluation panicked".to_string())
}

pub fn total_cost(graph: &Graph) -> usize { graph.ops.iter().map(|op| flop_cost(op, graph)).sum() } 