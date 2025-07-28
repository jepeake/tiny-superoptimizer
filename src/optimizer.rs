use crate::egraph::{EGraph, TensorOp, ENodeId, ShapeInfo};
use crate::op::OpKind;
use crate::graph::Graph;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Pattern {
    pub root: PatternNode,
}

#[derive(Debug, Clone)]
pub enum PatternNode {
    Op(TensorOp, Vec<PatternNode>),
    Var(String),
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct RewriteRule {
    pub name: String,
    pub lhs: Pattern,
    pub rhs: Pattern,
}

pub struct Optimizer {
    pub rules: Vec<RewriteRule>,
    pub max_iterations: usize,
}

impl Optimizer {
    pub fn new() -> Self {
        Optimizer {
            rules: Self::create_rules(),
            max_iterations: 20,
        }
    }

    fn create_rules() -> Vec<RewriteRule> {
        let mut rules = Vec::new();
        
        // High-Priority Rules
        
        // Self-Addition: x + x = x * 2 
        rules.push(RewriteRule {
            name: "self-add".to_string(),
            lhs: Pattern::binary_op("add", "a", "a"),
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Var("a".to_string()),
                        PatternNode::Op(TensorOp::Constant("2.0".to_string()), vec![])
                    ]
                )
            },
        });

        // Double Addition: (a + a) * b
        rules.push(RewriteRule {
            name: "double-add-mul".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("a".to_string())]),
                        PatternNode::Var("b".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Constant("2.0".to_string()), vec![]),
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Var("a".to_string()),
                                PatternNode::Var("b".to_string())
                            ]
                        )
                    ]
                )
            },
        });

        // Reciprocal Multiplication: a * (1/a) = 1 and (1/a) * a = 1
        for (name, lhs_pattern) in [
            ("mul-recip-improved", ("a", "recip_a")),
            ("recip-mul-improved", ("recip_a", "a"))
        ] {
            let (first, _second) = lhs_pattern;
            let (first_node, second_node) = if first == "a" {
                (PatternNode::Var("a".to_string()), PatternNode::Op(TensorOp::Recip(0), vec![PatternNode::Var("a".to_string())]))
            } else {
                (PatternNode::Op(TensorOp::Recip(0), vec![PatternNode::Var("a".to_string())]), PatternNode::Var("a".to_string()))
            };
            
            rules.push(RewriteRule {
                name: name.to_string(),
                lhs: Pattern {
                    root: PatternNode::Op(TensorOp::Mul(0, 0), vec![first_node, second_node])
                },
                rhs: Pattern::constant("1"),
            });
        }

        // Factor Common Terms: x * a + x * b → x * (a + b)
        rules.push(RewriteRule {
            name: "factor-common-mul".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Var("x".to_string()),
                                PatternNode::Var("a".to_string())
                            ]
                        ),
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Var("x".to_string()),
                                PatternNode::Var("b".to_string())
                            ]
                        )
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Var("x".to_string()),
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())])
                    ]
                )
            },
        });

        // Common Subexpression: exp(x) * a + exp(x) * b → exp(x) * (a + b)
        rules.push(RewriteRule {
            name: "factor-common-exp".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Op(TensorOp::Exp(0), vec![PatternNode::Var("x".to_string())]),
                                PatternNode::Var("a".to_string())
                            ]
                        ),
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Op(TensorOp::Exp(0), vec![PatternNode::Var("x".to_string())]),
                                PatternNode::Var("b".to_string())
                            ]
                        )
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Exp(0), vec![PatternNode::Var("x".to_string())]),
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())])
                    ]
                )
            },
        });
        
        // Commutative Rules
        rules.push(RewriteRule {
            name: "add-comm".to_string(),
            lhs: Pattern::binary_op("add", "a", "b"),
            rhs: Pattern::binary_op("add", "b", "a"),
        });
        
        rules.push(RewriteRule {
            name: "mul-comm".to_string(),
            lhs: Pattern::binary_op("mul", "a", "b"),
            rhs: Pattern::binary_op("mul", "b", "a"),
        });

        // Associativity Rules
        rules.push(RewriteRule {
            name: "add-assoc-left".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("c".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Var("a".to_string()),
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("c".to_string())])
                    ]
                )
            },
        });

        rules.push(RewriteRule {
            name: "mul-assoc-left".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("c".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Var("a".to_string()),
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("c".to_string())])
                    ]
                )
            },
        });

        // Distributivity Rules
        rules.push(RewriteRule {
            name: "distribute-left".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("c".to_string())]),
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("c".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("c".to_string())
                    ]
                )
            },
        });

        rules.push(RewriteRule {
            name: "distribute-right".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("c".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("c".to_string())]),
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("c".to_string())])
                    ]
                )
            },
        });

        // Mathematical Identities
        
        // Exponential/Logarithm Identities
        rules.push(RewriteRule {
            name: "exp-log".to_string(),
            lhs: Pattern::unary_op("exp", Pattern::unary_op("log", "a")),
            rhs: Pattern::var("a"),
        });

        rules.push(RewriteRule {
            name: "log-exp".to_string(),
            lhs: Pattern::unary_op("log", Pattern::unary_op("exp", "a")),
            rhs: Pattern::var("a"),
        });

        // Multiplicative Identity: 1 * x = x and x * 1 = x
        for (name, left, right) in [
            ("mul-identity-left", "1", "a"),
            ("mul-identity-right", "a", "1")
        ] {
            let left_pattern = if left == "1" { Pattern::constant("1") } else { Pattern::var("a") };
            let right_pattern = if right == "1" { Pattern::constant("1") } else { Pattern::var("a") };
            
            rules.push(RewriteRule {
                name: name.to_string(),
                lhs: Pattern::binary_op("mul", left_pattern, right_pattern),
                rhs: Pattern::var("a"),
            });
        }

        // Zero Multiplication: x * 0 = 0 and 0 * x = 0
        for (name, left, right) in [
            ("mul-zero-left", "a", "0"),
            ("mul-zero-right", "0", "a")
        ] {
            let left_pattern = if left == "0" { Pattern::constant("0") } else { Pattern::var("a") };
            let right_pattern = if right == "0" { Pattern::constant("0") } else { Pattern::var("a") };
            
            rules.push(RewriteRule {
                name: name.to_string(),
                lhs: Pattern::binary_op("mul", left_pattern, right_pattern),
                rhs: Pattern::constant("0"),
            });
        }

        // Min/Max: min(x,x) = x, max(x,x) = x  
        rules.push(RewriteRule {
            name: "max-self".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::MaxReduce(0, vec![]),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("a".to_string())])
                    ]
                )
            },
            rhs: Pattern::var("a"),
        });

        // Additive Identity: 0 + x = x and x + 0 = x
        for (name, left, right) in [
            ("add-identity-left", "0", "a"),
            ("add-identity-right", "a", "0")
        ] {
            let left_pattern = if left == "0" { Pattern::constant("0") } else { Pattern::var("a") };
            let right_pattern = if right == "0" { Pattern::constant("0") } else { Pattern::var("a") };
            
            rules.push(RewriteRule {
                name: name.to_string(),
                lhs: Pattern::binary_op("add", left_pattern, right_pattern),
                rhs: Pattern::var("a"),
            });
        }
        
        // Square Root: sqrt(a * a) = a
        rules.push(RewriteRule {
            name: "sqrt-square".to_string(),
            lhs: Pattern::unary_op("sqrt", Pattern::binary_op("mul", "a", "a")),
            rhs: Pattern::var("a"), 
        });
        
        // Broadcast Fusion: broadcast(a) + broadcast(b) → broadcast(a + b)
        rules.push(RewriteRule {
            name: "broadcast-add-fusion".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Broadcast(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("target".to_string())]),
                        PatternNode::Op(TensorOp::Broadcast(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("target".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Broadcast(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("target".to_string())
                    ]
                )
            },
        });

        rules.push(RewriteRule {
            name: "broadcast-mul-fusion".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Broadcast(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("target".to_string())]),
                        PatternNode::Op(TensorOp::Broadcast(0, 0), vec![PatternNode::Var("b".to_string()), PatternNode::Var("target".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Broadcast(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())]),
                        PatternNode::Var("target".to_string())
                    ]
                )
            },
        });
        
        // Sum Reduction Fusion Rules
        rules.push(RewriteRule {
            name: "sum-reduce-add-fusion".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::SumReduce(0, vec![]),
                    vec![
                        PatternNode::Op(TensorOp::Add(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Add(0, 0),
                    vec![
                        PatternNode::Op(TensorOp::SumReduce(0, vec![]), vec![PatternNode::Var("a".to_string())]),
                        PatternNode::Op(TensorOp::SumReduce(0, vec![]), vec![PatternNode::Var("b".to_string())])
                    ]
                )
            },
        });


        // Broadcast/Reduction Interaction Rules
        
        // Broadcast-SumReduce Pushdown: Enables Extracting Common Computations
        rules.push(RewriteRule {
            name: "broadcast-sum-reduce-pushdown".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::SumReduce(0, vec![]),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Op(TensorOp::Broadcast(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("target".to_string())]),
                                PatternNode::Var("b".to_string())
                            ]
                        ),
                        PatternNode::Var("axis".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Var("a".to_string()),
                        PatternNode::Op(
                            TensorOp::SumReduce(0, vec![]),
                            vec![
                                PatternNode::Var("b".to_string()),
                                PatternNode::Var("axis".to_string())
                            ]
                        )
                    ]
                )
            },
        });
        
        // Matrix Multiplication Associativity Rules

        rules.push(RewriteRule {
            name: "sum-reduce-mul-assoc".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::SumReduce(0, vec![]),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Var("a".to_string()),
                                PatternNode::Op(
                                    TensorOp::SumReduce(0, vec![]),
                                    vec![
                                        PatternNode::Op(
                                            TensorOp::Mul(0, 0),
                                            vec![
                                                PatternNode::Var("b".to_string()),
                                                PatternNode::Var("c".to_string())
                                            ]
                                        ),
                                        PatternNode::Var("axis2".to_string())
                                    ]
                                )
                            ]
                        ),
                        PatternNode::Var("axis1".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::SumReduce(0, vec![]),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Op(
                                    TensorOp::SumReduce(0, vec![]),
                                    vec![
                                        PatternNode::Op(
                                            TensorOp::Mul(0, 0),
                                            vec![
                                                PatternNode::Var("a".to_string()),
                                                PatternNode::Var("b".to_string())
                                            ]
                                        ),
                                        PatternNode::Var("axis1".to_string())
                                    ]
                                ),
                                PatternNode::Var("c".to_string())
                            ]
                        ),
                        PatternNode::Var("axis2".to_string())
                    ]
                )
            },
        });
        
        // Broadcast-Broadcast-Mul Distribution
        rules.push(RewriteRule {
            name: "broadcast-mul-distribution".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Broadcast(0, 0),
                    vec![
                        PatternNode::Op(
                            TensorOp::Mul(0, 0),
                            vec![
                                PatternNode::Var("a".to_string()),
                                PatternNode::Op(
                                    TensorOp::Mul(0, 0),
                                    vec![
                                        PatternNode::Var("b".to_string()),
                                        PatternNode::Var("c".to_string())
                                    ]
                                )
                            ]
                        ),
                        PatternNode::Var("target".to_string())
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Op(
                            TensorOp::Broadcast(0, 0),
                            vec![
                                PatternNode::Op(
                                    TensorOp::Mul(0, 0),
                                    vec![
                                        PatternNode::Var("a".to_string()),
                                        PatternNode::Var("b".to_string())
                                    ]
                                ),
                                PatternNode::Var("target".to_string())
                            ]
                        ),
                        PatternNode::Op(
                            TensorOp::Broadcast(0, 0),
                            vec![
                                PatternNode::Var("c".to_string()),
                                PatternNode::Var("target".to_string())
                            ]
                        )
                    ]
                )
            },
        });

        // Matrix Operation Fusion Rules
        
        // Reshape after Multiplication: reshape(a * b) → a * reshape(b)
        rules.push(RewriteRule {
            name: "mul-reshape-fusion".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Reshape(0, vec![]),
                    vec![
                        PatternNode::Op(TensorOp::Mul(0, 0), vec![PatternNode::Var("a".to_string()), PatternNode::Var("b".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Mul(0, 0),
                    vec![
                        PatternNode::Var("a".to_string()),
                        PatternNode::Op(TensorOp::Reshape(0, vec![]), vec![PatternNode::Var("b".to_string())])
                    ]
                )
            },
        });

        // Double Reshape: reshape(reshape(x, s1), s2) → reshape(x, s2)
        rules.push(RewriteRule {
            name: "double-reshape-elimination".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(
                    TensorOp::Reshape(0, vec![]),
                    vec![
                        PatternNode::Op(TensorOp::Reshape(0, vec![]), vec![PatternNode::Var("a".to_string())])
                    ]
                )
            },
            rhs: Pattern {
                root: PatternNode::Op(TensorOp::Reshape(0, vec![]), vec![PatternNode::Var("a".to_string())])
            },
        });

        // Identity Reshape: reshape(x, same_shape) → x
        rules.push(RewriteRule {
            name: "reshape-identity".to_string(),
            lhs: Pattern {
                root: PatternNode::Op(TensorOp::Reshape(0, vec![]), vec![PatternNode::Var("a".to_string())])
            },
            rhs: Pattern::var("a"),
        });
    }

    pub fn optimize(&mut self, egraph: &mut EGraph, root_id: ENodeId) -> ENodeId {
        let _initial_eclasses = egraph.eclasses.len();
        let mut prev_eclass_count = egraph.eclasses.len();
        let mut stagnant_iterations = 0;
        
        for _iteration in 0..self.max_iterations {
            let mut changed = false;
            let rules_to_apply: Vec<_> = self.rules.iter().cloned().collect();
            
            for rule in rules_to_apply {
                let rule_changed = self.apply_rule(egraph, &rule);
                changed |= rule_changed;
            }
            
            egraph.rebuild();
            
            // Check for Stagnation
            let current_eclass_count = egraph.eclasses.len();
            if current_eclass_count == prev_eclass_count {
                stagnant_iterations += 1;
                if stagnant_iterations >= 3 {
                    break; 
                }
            } else {
                stagnant_iterations = 0;
                prev_eclass_count = current_eclass_count;
            }
            
            if !changed {
                break;
            }
            
            // Safety Limit on E-Graph Size
            if egraph.eclasses.len() > 10000 {
                break;
            }
        }
        
        egraph.find(root_id)
    }

    fn apply_rule(&mut self, egraph: &mut EGraph, rule: &RewriteRule) -> bool {
        let mut changed = false;
        let eclass_ids: Vec<_> = (0..egraph.eclasses.len()).collect();
        
        for eclass_id in eclass_ids {
            let eclass = egraph.eclasses[eclass_id].clone();
            
            for op in &eclass.nodes {
                if let Some(matches) = self.pattern_match(&rule.lhs, op, egraph, eclass_id) {
                    if let Some(replacement_id) = self.instantiate_pattern(&rule.rhs, &matches, egraph) {
                        egraph.union(eclass_id, replacement_id);
                        changed = true;
                    }
                }
            }
        }
        
        changed
    }

    fn pattern_match(&self, pattern: &Pattern, op: &TensorOp, egraph: &EGraph, eclass_id: ENodeId) -> Option<HashMap<String, ENodeId>> {
        let mut matches = HashMap::new();
        if self.match_node(&pattern.root, op, &mut matches, egraph, eclass_id) {
            Some(matches)
        } else {
            None
        }
    }

    fn match_node(&self, pattern_node: &PatternNode, op: &TensorOp, matches: &mut HashMap<String, ENodeId>, egraph: &EGraph, current_eclass_id: ENodeId) -> bool {
        match pattern_node {
            PatternNode::Var(name) => {
                if let Some(&existing_id) = matches.get(name) {
                    // Check if Current EClass is Equivalent to the Existing Match
                    existing_id == current_eclass_id
                } else {
                    matches.insert(name.clone(), current_eclass_id);
                    true
                }
            },
            PatternNode::Wildcard => true,
            PatternNode::Op(pattern_op, pattern_children) => {
                if !self.ops_structurally_equal(pattern_op, op) {
                    return false;
                }
                
                let op_children = op.get_children();
                if pattern_children.len() != op_children.len() {
                    return false;
                }
                
                // For Each Child, Recursively Match in the EGraph
                for (pattern_child, &op_child_id) in pattern_children.iter().zip(op_children.iter()) {
                    // OpChildId is the EClass ID for This Child
                    let child_eclass_id = op_child_id;
                    
                    // Try to Match the Pattern Against Any Node in the Child's EClass
                    if let Some(child_eclass) = egraph.eclasses.get(child_eclass_id) {
                        let mut found_match = false;
                        
                        // Try to Match Against Each Node in the Child EClass
                        for child_op in &child_eclass.nodes {
                            let mut temp_matches = matches.clone();
                            if self.match_node(pattern_child, child_op, &mut temp_matches, egraph, child_eclass_id) {
                                *matches = temp_matches;
                                found_match = true;
                                break;
                            }
                        }
                        
                        if !found_match {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                
                true
            }
        }
    }

    fn ops_structurally_equal(&self, op1: &TensorOp, op2: &TensorOp) -> bool {
        std::mem::discriminant(op1) == std::mem::discriminant(op2)
    }

    fn instantiate_pattern(&self, pattern: &Pattern, matches: &HashMap<String, ENodeId>, egraph: &mut EGraph) -> Option<ENodeId> {
        self.instantiate_node(&pattern.root, matches, egraph)
    }

    fn instantiate_node(&self, pattern_node: &PatternNode, matches: &HashMap<String, ENodeId>, egraph: &mut EGraph) -> Option<ENodeId> {
        match pattern_node {
            PatternNode::Var(name) => matches.get(name).copied(),
            PatternNode::Wildcard => None,
            PatternNode::Op(op_template, pattern_children) => {
                let mut child_ids = Vec::new();
                for pattern_child in pattern_children {
                    if let Some(child_id) = self.instantiate_node(pattern_child, matches, egraph) {
                        child_ids.push(child_id);
                    } else {
                        return None;
                    }
                }
                
                let new_op = op_template.replace_children(&child_ids);
                Some(egraph.add(new_op))
            }
        }
    }
}

impl Pattern {
    pub fn var(name: &str) -> Pattern {
        Pattern {
            root: PatternNode::Var(name.to_string())
        }
    }

    pub fn constant(value: &str) -> Pattern {
        Pattern {
            root: PatternNode::Op(
                TensorOp::Constant(value.to_string()),
                vec![]
            )
        }
    }

    pub fn binary_op(op: &str, left: impl Into<PatternNode>, right: impl Into<PatternNode>) -> Pattern {
        let op_template = match op {
            "add" => TensorOp::Add(0, 0),
            "mul" => TensorOp::Mul(0, 0),
            "mod" => TensorOp::Mod(0, 0),
            "lessthan" => TensorOp::LessThan(0, 0),
            "broadcast" => TensorOp::Broadcast(0, 0),
            _ => TensorOp::Add(0, 0),
        };
        
        Pattern {
            root: PatternNode::Op(op_template, vec![left.into(), right.into()])
        }
    }

    pub fn unary_op(op: &str, arg: impl Into<PatternNode>) -> Pattern {
        let op_template = match op {
            "sin" => TensorOp::Sin(0),
            "sqrt" => TensorOp::Sqrt(0),
            "recip" => TensorOp::Recip(0),
            "exp" => TensorOp::Exp(0),
            "log" => TensorOp::Log(0),
            "sum" => TensorOp::SumReduce(0, vec![]),
            "reshape" => TensorOp::Reshape(0, vec![]),
            _ => TensorOp::Reshape(0, vec![]),
        };
        
        Pattern {
            root: PatternNode::Op(op_template, vec![arg.into()])
        }
    }

    pub fn ternary_op(op: &str, arg1: impl Into<PatternNode>, arg2: impl Into<PatternNode>, arg3: impl Into<PatternNode>) -> Pattern {
        let op_template = match op {
            "constant" => TensorOp::Constant("0.0".to_string()),
            _ => TensorOp::Constant("0.0".to_string()),
        };
        
        Pattern {
            root: PatternNode::Op(op_template, vec![arg1.into(), arg2.into(), arg3.into()])
        }
    }
}

impl From<&str> for PatternNode {
    fn from(s: &str) -> Self {
        PatternNode::Var(s.to_string())
    }
}

impl From<Pattern> for PatternNode {
    fn from(p: Pattern) -> Self {
        p.root
    }
}

pub fn graph_to_egraph(graph: &Graph) -> (EGraph, ENodeId) {
    let mut egraph = EGraph::new();
    let mut node_map = HashMap::new();
    
    for (i, tensor) in graph.tensors.iter().enumerate() {
        let shape_info = ShapeInfo::new(tensor.shape.clone(), vec![]);
        let id = egraph.add(TensorOp::Symbol(tensor.name.clone(), shape_info));
        node_map.insert(i, id);
    }
    
    for (i, op) in graph.ops.iter().enumerate() {
        let tensor_idx = graph.tensors.len() - graph.ops.len() + i;
        let tensor_op = match op.kind {
            OpKind::Add => TensorOp::Add(node_map[&op.inputs[0]], node_map[&op.inputs[1]]),
            OpKind::Mul => TensorOp::Mul(node_map[&op.inputs[0]], node_map[&op.inputs[1]]),
            OpKind::SumReduce => TensorOp::SumReduce(node_map[&op.inputs[0]], op.attrs.clone()),
            OpKind::MaxReduce => TensorOp::MaxReduce(node_map[&op.inputs[0]], op.attrs.clone()),
            OpKind::Reshape => TensorOp::Reshape(node_map[&op.inputs[0]], op.attrs.clone()),
            OpKind::Broadcast => TensorOp::Broadcast(node_map[&op.inputs[0]], node_map[&op.inputs[1]]),
            OpKind::Sin => TensorOp::Sin(node_map[&op.inputs[0]]),
            OpKind::Sqrt => TensorOp::Sqrt(node_map[&op.inputs[0]]),
            OpKind::Recip => TensorOp::Recip(node_map[&op.inputs[0]]),
            OpKind::Exp => TensorOp::Exp(node_map[&op.inputs[0]]),
            OpKind::Log => TensorOp::Log(node_map[&op.inputs[0]]),
            OpKind::Mod => TensorOp::Mod(node_map[&op.inputs[0]], node_map[&op.inputs[1]]),
            OpKind::LessThan => TensorOp::LessThan(node_map[&op.inputs[0]], node_map[&op.inputs[1]]),
            OpKind::Constant => TensorOp::Constant("1.0".to_string()),
            OpKind::Input(ref name) => {
                let shape = if let Some(tensor) = graph.tensors.iter().find(|t| t.name == *name) {
                    tensor.shape.clone()
                } else {
                    vec![]
                };
                TensorOp::Symbol(name.clone(), ShapeInfo::new(shape, vec![]))
            },
        };
        let id = egraph.add(tensor_op);
        node_map.insert(tensor_idx, id);
    }
    
    let root_id = if graph.outputs.is_empty() {
        0
    } else {
        node_map[&graph.outputs[0]]
    };
    
    (egraph, root_id)
}

pub fn egraph_to_graph(egraph: &mut EGraph, root_id: ENodeId, original: &Graph) -> Graph {
    let best_expr = egraph.extract_best_tree(root_id);
    tensor_op_to_graph(&best_expr, original)
}

fn tensor_op_to_graph(expr: &TensorOp, original: &Graph) -> Graph {
    let mut new_graph = Graph::new();
    let mut symbol_map = HashMap::new();
    let mut memo = HashMap::new();
    
    let input_tensor_count = original.tensors.len() - original.ops.len();
    for (i, tensor) in original.tensors.iter().take(input_tensor_count).enumerate() {
        let tensor_id = new_graph.add_tensor(tensor.shape.clone(), tensor.name.clone());
        symbol_map.insert(tensor.name.clone(), tensor_id);
        // Verify the Tensor Gets the Expected ID
        assert_eq!(tensor_id, i, "Input tensor {} got wrong ID: expected {}, got {}", tensor.name, i, tensor_id);
    }
    
    let result_id = build_graph_recursive(expr, &mut new_graph, &mut symbol_map, &mut memo, original);
    new_graph.outputs.push(result_id);
    new_graph
}


fn build_graph_recursive(
    expr: &TensorOp, 
    graph: &mut Graph, 
    symbol_map: &mut HashMap<String, usize>,
    memo: &mut HashMap<TensorOp, usize>,
    original: &Graph
) -> usize {
    use crate::op::OpKind;
    
    // Check Memo to Avoid Rebuilding the Same Expression
    if let Some(&cached_id) = memo.get(expr) {
        return cached_id;
    }
    
    let result_id = match expr {
        TensorOp::Symbol(name, _) => {
            if let Some(&tensor_id) = symbol_map.get(name) {
                tensor_id
            } else {
                // Find the Tensor in Original Graph to Get Correct Shape
                if let Some(original_tensor) = original.tensors.iter().find(|t| t.name == *name) {
                    let tensor_id = graph.add_tensor(original_tensor.shape.clone(), name.clone());
                    symbol_map.insert(name.clone(), tensor_id);
                    tensor_id
                } else {
                    // Fallback: Create a Placeholder Tensor
                    let tensor_id = graph.add_tensor(vec![1], name.clone());
                    symbol_map.insert(name.clone(), tensor_id);
                    tensor_id
                }
            }
        },
        
        // Tree Variants - Recursively Build Children
        TensorOp::AddTree(a, b) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            let input_b = build_graph_recursive(b, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Add, vec![input_a, input_b], vec![])
        },
        TensorOp::MulTree(a, b) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            let input_b = build_graph_recursive(b, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Mul, vec![input_a, input_b], vec![])
        },
        TensorOp::SumReduceTree(a, axes) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::SumReduce, vec![input_a], axes.clone())
        },
        TensorOp::SinTree(a) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Sin, vec![input_a], vec![])
        },
        TensorOp::ExpTree(a) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Exp, vec![input_a], vec![])
        },
        TensorOp::LogTree(a) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Log, vec![input_a], vec![])
        },
        TensorOp::ModTree(a, b) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            let input_b = build_graph_recursive(b, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Mod, vec![input_a, input_b], vec![])
        },
        TensorOp::LessThanTree(a, b) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            let input_b = build_graph_recursive(b, graph, symbol_map, memo, original);
            graph.add_op(OpKind::LessThan, vec![input_a, input_b], vec![])
        },
        TensorOp::RecipTree(a) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Recip, vec![input_a], vec![])
        },
        TensorOp::SqrtTree(a) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Sqrt, vec![input_a], vec![])
        },
        TensorOp::ReshapeTree(a, shape) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Reshape, vec![input_a], shape.clone())
        },
        TensorOp::BroadcastTree(a, b) => {
            let input_a = build_graph_recursive(a, graph, symbol_map, memo, original);
            let input_b = build_graph_recursive(b, graph, symbol_map, memo, original);
            graph.add_op(OpKind::Broadcast, vec![input_a, input_b], vec![])
        },
        TensorOp::ConstantTree(_value) => {
            graph.add_op(OpKind::Constant, vec![], vec![])
        },
        
        // Non-Tree Variants (Shouldn't Appear in Extracted Tree, But Handle for Completeness)
        _ => {
            // This Shouldn't Happen if We're Using ExtractBestTree Properly
            // Fallback to Creating a Dummy Tensor
            if symbol_map.is_empty() {
                graph.add_tensor(vec![1], "fallback".to_string())
            } else {
                *symbol_map.values().next().unwrap()
            }
        }
    };
    
    memo.insert(expr.clone(), result_id);
    result_id
}

pub fn optimize_with_egraph(graph: &Graph) -> Graph {
    // Convert Graph to EGraph
    let (mut egraph, root_id) = graph_to_egraph(graph);
    
    // Create Optimizer and Run Optimization
    let mut optimizer = Optimizer::new();
    let optimized_root = optimizer.optimize(&mut egraph, root_id);
    
    // Convert Back to Graph
    egraph_to_graph(&mut egraph, optimized_root, graph)
}


impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}