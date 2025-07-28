#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    // Unary
    Exp,
    Log,
    Sin,
    Recip,        
    Sqrt,

    // Binary
    Add,
    Mul,
    Mod,
    LessThan,
    
    // Reduce
    SumReduce,    
    MaxReduce,    
    
    // Broadcast
    Broadcast,    
    Reshape,      
    
    // Constant
    Constant,   

    // Symbol
    Input(String),
}

#[derive(Clone, Debug)]
pub struct Op {
    pub kind: OpKind,
    pub inputs: Vec<usize>,
    pub attrs: Vec<i64>,
} 