use std::collections::{HashMap, HashSet};
use std::fmt;
pub type ENodeId = usize;

/// Shape & Attribute Information for Tensor Operations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeInfo {
    pub shape: Vec<usize>,
    pub attrs: Vec<i64>,  // For Storing Reduction Axes, Reshape Dims, etc.
}

impl ShapeInfo {
    pub fn new(shape: Vec<usize>, attrs: Vec<i64>) -> Self { Self { shape, attrs } }
    pub fn unknown() -> Self { Self { shape: vec![], attrs: vec![] } }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorOp {
    Exp(ENodeId), Log(ENodeId), Sin(ENodeId), Recip(ENodeId), Sqrt(ENodeId),
    Add(ENodeId, ENodeId), Mul(ENodeId, ENodeId), Mod(ENodeId, ENodeId), LessThan(ENodeId, ENodeId),
    SumReduce(ENodeId, Vec<i64>), MaxReduce(ENodeId, Vec<i64>),
    Broadcast(ENodeId, ENodeId), Reshape(ENodeId, Vec<i64>),
    Constant(String), Symbol(String, ShapeInfo),
    
    // Tree Variants for Reconstruction
    ExpTree(Box<TensorOp>),
    LogTree(Box<TensorOp>),
    SinTree(Box<TensorOp>),
    RecipTree(Box<TensorOp>),
    SqrtTree(Box<TensorOp>),
    AddTree(Box<TensorOp>, Box<TensorOp>),
    MulTree(Box<TensorOp>, Box<TensorOp>),
    ModTree(Box<TensorOp>, Box<TensorOp>),
    LessThanTree(Box<TensorOp>, Box<TensorOp>),
    SumReduceTree(Box<TensorOp>, Vec<i64>),
    MaxReduceTree(Box<TensorOp>, Vec<i64>),
    BroadcastTree(Box<TensorOp>, Box<TensorOp>),
    ReshapeTree(Box<TensorOp>, Vec<i64>),
    ConstantTree(String),
}

impl fmt::Display for TensorOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorOp::Exp(a) => write!(f, "(exp {})", a),
            TensorOp::Log(a) => write!(f, "(log {})", a),
            TensorOp::Sin(a) => write!(f, "(sin {})", a),
            TensorOp::Recip(a) => write!(f, "(recip {})", a),
            TensorOp::Sqrt(a) => write!(f, "(sqrt {})", a),
            TensorOp::Add(a, b) => write!(f, "(+ {} {})", a, b),
            TensorOp::Mul(a, b) => write!(f, "(* {} {})", a, b),
            TensorOp::Mod(a, b) => write!(f, "(mod {} {})", a, b),
            TensorOp::LessThan(a, b) => write!(f, "(< {} {})", a, b),

            TensorOp::SumReduce(a, axes) => if axes.is_empty() { write!(f, "(sum_reduce {})", a) } else { write!(f, "(sum_reduce {} axis={:?})", a, axes) },
            TensorOp::MaxReduce(a, axes) => if axes.is_empty() { write!(f, "(max_reduce {})", a) } else { write!(f, "(max_reduce {} axis={:?})", a, axes) },
            TensorOp::Broadcast(a, b) => write!(f, "(broadcast {} {})", a, b),
            TensorOp::Reshape(a, new_shape) => write!(f, "(reshape {} {:?})", a, new_shape),
            TensorOp::Constant(s) => write!(f, "(const {})", s),
            TensorOp::Symbol(s, _) => write!(f, "{}", s),
            
            // Tree Variants 
            TensorOp::ExpTree(a) => write!(f, "(exp {})", a),
            TensorOp::LogTree(a) => write!(f, "(log {})", a),
            TensorOp::SinTree(a) => write!(f, "(sin {})", a),
            TensorOp::RecipTree(a) => write!(f, "(recip {})", a),
            TensorOp::SqrtTree(a) => write!(f, "(sqrt {})", a),
            TensorOp::AddTree(a, b) => write!(f, "(+ {} {})", a, b),
            TensorOp::MulTree(a, b) => write!(f, "(* {} {})", a, b),
            TensorOp::ModTree(a, b) => write!(f, "(mod {} {})", a, b),
            TensorOp::LessThanTree(a, b) => write!(f, "(< {} {})", a, b),
            TensorOp::SumReduceTree(a, axes) => if axes.is_empty() { write!(f, "(sum_reduce {})", a) } else { write!(f, "(sum_reduce {} axis={:?})", a, axes) },
            TensorOp::MaxReduceTree(a, axes) => if axes.is_empty() { write!(f, "(max_reduce {})", a) } else { write!(f, "(max_reduce {} axis={:?})", a, axes) },
            TensorOp::BroadcastTree(a, b) => write!(f, "(broadcast {} {})", a, b),
            TensorOp::ReshapeTree(a, new_shape) => write!(f, "(reshape {} {:?})", a, new_shape),
            TensorOp::ConstantTree(s) => write!(f, "(const {})", s),
        }
    }
}

impl TensorOp {
    pub fn get_children(&self) -> Vec<ENodeId> {
        match self {
            // Binary
            TensorOp::Add(a, b) | TensorOp::Mul(a, b) | TensorOp::Mod(a, b) |
            TensorOp::LessThan(a, b) | TensorOp::Broadcast(a, b) => vec![*a, *b],
            
            // Unary
            TensorOp::Exp(a) | TensorOp::Log(a) | TensorOp::Sin(a) | TensorOp::Recip(a) |
            TensorOp::Sqrt(a) => vec![*a],
            TensorOp::SumReduce(a, _) | TensorOp::MaxReduce(a, _) | 
            TensorOp::Reshape(a, _) => vec![*a],
            
            // Leaf
            TensorOp::Symbol(_, _) | TensorOp::Constant(_) => vec![],
            
            // Tree Variants
            TensorOp::AddTree(_, _) | TensorOp::MulTree(_, _) | TensorOp::ModTree(_, _) |
            TensorOp::LessThanTree(_, _) | TensorOp::BroadcastTree(_, _) => vec![],
            TensorOp::ExpTree(_) | TensorOp::LogTree(_) | TensorOp::SinTree(_) | 
            TensorOp::RecipTree(_) | TensorOp::SqrtTree(_) | TensorOp::SumReduceTree(_, _) | 
            TensorOp::MaxReduceTree(_, _) | TensorOp::ReshapeTree(_, _) => vec![],
            TensorOp::ConstantTree(_) => vec![],
        }
    }

    pub fn replace_children(&self, new_children: &[ENodeId]) -> TensorOp {
        match self {
            // Binary
            TensorOp::Add(_, _) => TensorOp::Add(new_children[0], new_children[1]),
            TensorOp::Mul(_, _) => TensorOp::Mul(new_children[0], new_children[1]),
            TensorOp::Mod(_, _) => TensorOp::Mod(new_children[0], new_children[1]),
            TensorOp::LessThan(_, _) => TensorOp::LessThan(new_children[0], new_children[1]),
            TensorOp::Broadcast(_, _) => TensorOp::Broadcast(new_children[0], new_children[1]),
            
            // Unary
            TensorOp::Exp(_) => TensorOp::Exp(new_children[0]),
            TensorOp::Log(_) => TensorOp::Log(new_children[0]),
            TensorOp::Sin(_) => TensorOp::Sin(new_children[0]),
            TensorOp::Recip(_) => TensorOp::Recip(new_children[0]),
            TensorOp::Sqrt(_) => TensorOp::Sqrt(new_children[0]),
            TensorOp::SumReduce(_, axes) => TensorOp::SumReduce(new_children[0], axes.clone()),
            TensorOp::MaxReduce(_, axes) => TensorOp::MaxReduce(new_children[0], axes.clone()),
            TensorOp::Reshape(_, shape) => TensorOp::Reshape(new_children[0], shape.clone()),
            
            // Leaf
            TensorOp::Symbol(s, shape) => TensorOp::Symbol(s.clone(), shape.clone()),
            TensorOp::Constant(s) => TensorOp::Constant(s.clone()),
            tree_op => tree_op.clone(),
        }
    }

    pub fn cost(&self) -> usize {
        match self {
            // Op Costs
            TensorOp::Add(_, _) | TensorOp::AddTree(_, _) => 1,
            TensorOp::Mul(_, _) | TensorOp::MulTree(_, _) => 1,
            TensorOp::Mod(_, _) | TensorOp::ModTree(_, _) => 2,
            TensorOp::LessThan(_, _) | TensorOp::LessThanTree(_, _) => 1,
            TensorOp::Exp(_) | TensorOp::ExpTree(_) => 20,
            TensorOp::Log(_) | TensorOp::LogTree(_) => 25,
            TensorOp::Sin(_) | TensorOp::SinTree(_) => 15,
            TensorOp::Recip(_) | TensorOp::RecipTree(_) => 10,
            TensorOp::Sqrt(_) | TensorOp::SqrtTree(_) => 10,
            TensorOp::SumReduce(_, _) | TensorOp::SumReduceTree(_, _) => 5,
            TensorOp::MaxReduce(_, _) | TensorOp::MaxReduceTree(_, _) => 5,
            TensorOp::Broadcast(_, _) | TensorOp::BroadcastTree(_, _) => 1,
            TensorOp::Reshape(_, _) | TensorOp::ReshapeTree(_, _) => 0,
            TensorOp::Constant(_) | TensorOp::ConstantTree(_) => 0,
            TensorOp::Symbol(_, _) => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ENode {
    pub op: TensorOp,
    pub eclass_id: ENodeId,
}

#[derive(Debug, Clone)]
pub struct EClass {
    pub id: ENodeId,
    pub nodes: HashSet<TensorOp>,
    pub parents: HashSet<ENodeId>,
}

impl EClass {
    pub fn new(id: ENodeId) -> Self {
        Self { id, nodes: HashSet::new(), parents: HashSet::new() }
    }

    pub fn merge(&mut self, other: &EClass) {
        self.nodes.extend(other.nodes.iter().cloned());
        self.parents.extend(other.parents.iter().cloned());
    }
}

#[derive(Debug, Clone)]
pub struct EGraph {
    pub eclasses: Vec<EClass>,
    pub hashcons: HashMap<TensorOp, ENodeId>,
    pub unionfind: Vec<ENodeId>,
}

impl EGraph {
    pub fn new() -> Self {
        Self { eclasses: Vec::new(), hashcons: HashMap::new(), unionfind: Vec::new() }
    }

    pub fn add(&mut self, op: TensorOp) -> ENodeId {
        let canonical_op = self.canonicalize_op(&op);
        
        if let Some(&existing_id) = self.hashcons.get(&canonical_op) {
            return self.find(existing_id);
        }

        let id = self.eclasses.len();
        let mut eclass = EClass::new(id);
        eclass.nodes.insert(canonical_op.clone());
        
        self.eclasses.push(eclass);
        self.unionfind.push(id);
        self.hashcons.insert(canonical_op, id);

        for child_id in op.get_children() {
            let child_eclass_id = self.find(child_id);
            if child_eclass_id < self.eclasses.len() {
                self.eclasses[child_eclass_id].parents.insert(id);
            }
        }

        id
    }

    fn canonicalize_op(&mut self, op: &TensorOp) -> TensorOp {
        let children = op.get_children();
        let canonical_children: Vec<_> = children.iter().map(|&id| self.find(id)).collect();
        if children == canonical_children { op.clone() } else { op.replace_children(&canonical_children) }
    }

    pub fn find(&mut self, mut id: ENodeId) -> ENodeId {
        let mut root = id;
        while self.unionfind[root] != root {
            root = self.unionfind[root];
        }

        while self.unionfind[id] != root {
            let next = self.unionfind[id];
            self.unionfind[id] = root;
            id = next;
        }

        root
    }

    pub fn union(&mut self, id1: ENodeId, id2: ENodeId) -> ENodeId {
        let root1 = self.find(id1);
        let root2 = self.find(id2);

        if root1 == root2 {
            return root1;
        }

        let (smaller, larger) = if self.eclasses[root1].nodes.len() < self.eclasses[root2].nodes.len() { (root1, root2) } else { (root2, root1) };

        self.unionfind[smaller] = larger;

        let smaller_eclass = self.eclasses[smaller].clone();
        self.eclasses[larger].merge(&smaller_eclass);

        let parents_to_update: Vec<ENodeId> = smaller_eclass.parents.iter().cloned().collect();
        for parent_id in parents_to_update {
            self.repair_hashcons(parent_id);
        }

        larger
    }

    fn repair_hashcons(&mut self, eclass_id: ENodeId) {
        let eclass = self.eclasses[eclass_id].clone();
        let mut ops_to_rehash = Vec::new();

        for op in &eclass.nodes {
            ops_to_rehash.push(op.clone());
        }

        for op in ops_to_rehash {
            self.hashcons.remove(&op);
            let canonical_op = self.canonicalize_op(&op);
            self.hashcons.insert(canonical_op, eclass_id);
        }
    }

    pub fn rebuild(&mut self) {
        loop {
            let old_hashcons = self.hashcons.clone();
            self.hashcons.clear();

            let eclasses_copy = self.eclasses.clone();
            
            for (eclass_id, eclass) in eclasses_copy.iter().enumerate() {
                for op in &eclass.nodes {
                    let canonical_op = self.canonicalize_op(op);
                    if let Some(&existing_id) = self.hashcons.get(&canonical_op) {
                        if existing_id != eclass_id {
                            self.union(eclass_id, existing_id);
                        }
                    } else {
                        self.hashcons.insert(canonical_op, eclass_id);
                    }
                }
            }

            if self.hashcons.len() == old_hashcons.len() &&
                self.hashcons.iter().all(|(k, v)| old_hashcons.get(k) == Some(v)) {
                break;
            }
        }
    }

    pub fn get_best_expr(&mut self, root_id: ENodeId) -> TensorOp {
        let root_id = self.find(root_id);
        let mut memo = HashMap::new();
        self.extract_best_recursive(root_id, &mut memo)
    }

    pub fn extract_best_tree(&mut self, root_id: ENodeId) -> TensorOp {
        let root_id = self.find(root_id);
        let mut memo = HashMap::new();
        self.extract_tree_recursive(root_id, &mut memo)
    }

    fn extract_tree_recursive(&mut self, eclass_id: ENodeId, memo: &mut HashMap<ENodeId, TensorOp>) -> TensorOp {
        let eclass_id = self.find(eclass_id);

        if let Some(op) = memo.get(&eclass_id) {
            return op.clone();
        }

        let mut best_cost = usize::MAX;
        let mut best_op = None;

        // Convert HashSet to Vec and Sort for Deterministic Iteration
        let mut ops: Vec<_> = self.eclasses[eclass_id].nodes.iter().cloned().collect();
        ops.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

        for op in ops {
            let mut total_cost = op.cost();
            let mut new_op = op.clone();

            let children = op.get_children();
            if !children.is_empty() {
                let mut new_children = Vec::new();
                for &child_id in &children {
                    let child_op = self.extract_tree_recursive(child_id, memo);
                    total_cost += self.compute_tree_cost(&child_op);
                    new_children.push(child_op);
                }
                new_op = self.rebuild_op_with_tree_children(&op, &new_children);
                
                // Apply Structural Penalties for Inefficient Patterns
            }

            if total_cost < best_cost {
                best_cost = total_cost;
                best_op = Some(new_op);
            }
        }

        let result_op = best_op.unwrap_or_else(|| TensorOp::Symbol("error".to_string(), ShapeInfo::unknown()));
        memo.insert(eclass_id, result_op.clone());
        result_op
    }

    fn compute_tree_cost(&self, op: &TensorOp) -> usize {
        op.cost()
    }

    fn rebuild_op_with_tree_children(&self, op: &TensorOp, tree_children: &[TensorOp]) -> TensorOp {
        match op {
            // Binary
            TensorOp::Add(_, _) => TensorOp::AddTree(Box::new(tree_children[0].clone()), Box::new(tree_children[1].clone())),
            TensorOp::Mul(_, _) => TensorOp::MulTree(Box::new(tree_children[0].clone()), Box::new(tree_children[1].clone())),
            TensorOp::Mod(_, _) => TensorOp::ModTree(Box::new(tree_children[0].clone()), Box::new(tree_children[1].clone())),
            TensorOp::LessThan(_, _) => TensorOp::LessThanTree(Box::new(tree_children[0].clone()), Box::new(tree_children[1].clone())),
            TensorOp::Broadcast(_, _) => TensorOp::BroadcastTree(Box::new(tree_children[0].clone()), Box::new(tree_children[1].clone())),
            
            // Unary
            TensorOp::Exp(_) => TensorOp::ExpTree(Box::new(tree_children[0].clone())),
            TensorOp::Log(_) => TensorOp::LogTree(Box::new(tree_children[0].clone())),
            TensorOp::Sin(_) => TensorOp::SinTree(Box::new(tree_children[0].clone())),
            TensorOp::Recip(_) => TensorOp::RecipTree(Box::new(tree_children[0].clone())),
            TensorOp::Sqrt(_) => TensorOp::SqrtTree(Box::new(tree_children[0].clone())),
            TensorOp::SumReduce(_, axes) => TensorOp::SumReduceTree(Box::new(tree_children[0].clone()), axes.clone()),
            TensorOp::MaxReduce(_, axes) => TensorOp::MaxReduceTree(Box::new(tree_children[0].clone()), axes.clone()),
            TensorOp::Reshape(_, shape) => TensorOp::ReshapeTree(Box::new(tree_children[0].clone()), shape.clone()),
            
            // Leaf
            TensorOp::Symbol(s, shape) => TensorOp::Symbol(s.clone(), shape.clone()),
            TensorOp::Constant(s) => TensorOp::ConstantTree(s.clone()),
            
            // Tree Variants (Should Not Be Rebuilt Again)
            tree_op => tree_op.clone(),
        }
    }

    fn extract_best_recursive(&mut self, eclass_id: ENodeId, memo: &mut HashMap<ENodeId, (usize, TensorOp)>) -> TensorOp {
        let eclass_id = self.find(eclass_id);

        if let Some((_, op)) = memo.get(&eclass_id) {
            return op.clone();
        }

        let mut best_cost = usize::MAX;
        let mut best_op = None;

        for op in &self.eclasses[eclass_id].nodes.clone() {
            let mut total_cost = op.cost();
            let mut new_op = op.clone();

            let children = op.get_children();
            if !children.is_empty() {
                let mut new_children = Vec::new();
                for &child_id in &children {
                    let _child_op = self.extract_best_recursive(child_id, memo);
                    let (child_cost, _) = memo.get(&self.find(child_id)).unwrap();
                    total_cost += child_cost;
                    new_children.push(self.find(child_id));
                }
                new_op = op.replace_children(&new_children);
            }

            if total_cost < best_cost {
                best_cost = total_cost;
                best_op = Some(new_op);
            }
        }

        let result_op = best_op.unwrap_or_else(|| TensorOp::Symbol("error".to_string(), ShapeInfo::unknown()));
        memo.insert(eclass_id, (best_cost, result_op.clone()));
        result_op
    }
}

impl Default for EGraph {
    fn default() -> Self { Self::new() }
}