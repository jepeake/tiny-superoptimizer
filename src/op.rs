#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Exp, Log, Sin, Recip, Sqrt,
    Add, Mul, Mod, LessThan,
    SumReduce, MaxReduce,
    Broadcast, Reshape,
    Constant,
    Input(String),
}

#[derive(Clone, Debug)]
pub struct Op {
    pub kind: OpKind,
    pub inputs: Vec<usize>,
    pub attrs: Vec<i64>,
} 