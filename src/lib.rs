pub mod tensor;
pub mod op;
pub mod graph;
pub mod egraph;
pub mod optimizer;
pub mod eval;
pub mod visualization;

pub use tensor::*;
pub use op::*;
pub use graph::*;
pub use optimizer::*;
pub use eval::*;
pub use visualization::*;

pub fn benchmark_optimization(graph: &Graph, name: &str) {
    println!("=== {} ===", name);
    let original_cost = total_cost(graph);
    let original_ops = graph.ops.len();
    
    let optimized = optimize_with_egraph(graph);
    let optimized_cost = total_cost(&optimized);
    let optimized_ops = optimized.ops.len();
    
    let equiv = equivalent(graph, &optimized, 10);
    let speedup = original_cost as f64 / optimized_cost.max(1) as f64;
    
    println!("Original:  {} ops, cost {}", original_ops, original_cost);
    println!("Optimized: {} ops, cost {}", optimized_ops, optimized_cost);
    println!("Speedup: {:.2}x | Verified: {}", speedup, if equiv { "YES" } else { "NO" });
    println!();
}

