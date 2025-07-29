use crate::graph::Graph;
use crate::op::{OpKind};
use plotters::prelude::*;
use plotters_bitmap::BitMapBackend;

fn clean_name(name: &str) -> String {
    name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        .replace("*", "mul").replace("→", "to").replace("/", "_").replace(":", "_")
        .replace("'", "").replace("\"", "").replace("&", "and").to_lowercase()
}

#[derive(Clone)]
struct GraphNode { id: usize, name: String, shape: Vec<usize>, node_type: NodeType, x: i32, y: i32 }

#[derive(Clone)]
enum NodeType { Input, Output, Operation(OpKind, usize) }

fn layout_graph(graph: &Graph) -> (Vec<GraphNode>, Vec<(usize, usize)>) {
    let mut nodes = Vec::new();
    
    let layers = create_layered_layout(graph);
    
    let positioned_nodes = position_nodes_in_layers(&layers, graph);
    
    for (node_id, (x, y, node_info)) in positioned_nodes {
        let graph_node = match node_info {
            NodeInfo::Input(tensor_idx) => {
                let tensor = &graph.tensors[tensor_idx];
                GraphNode {
                    id: node_id,
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    node_type: NodeType::Input,
                    x,
                    y,
                }
            }
            NodeInfo::Operation(op_idx) => {
                let op = &graph.ops[op_idx];
                let cost = crate::eval::flop_cost(op, graph);
                GraphNode {
                    id: node_id,
                    name: format!("op{}", op_idx),
                    shape: vec![],
                    node_type: NodeType::Operation(op.kind.clone(), cost),
                    x,
                    y,
                }
            }
            NodeInfo::Output => {
                GraphNode {
                    id: node_id,
                    name: "result".to_string(),
                    shape: vec![16, 16],
                    node_type: NodeType::Output,
                    x,
                    y,
                }
            }
        };
        nodes.push(graph_node);
    }
    
    let edges = create_clean_edges(graph);
    
    (nodes, edges)
}

#[derive(Clone)]
enum NodeInfo { Input(usize), Operation(usize), Output }

fn create_layered_layout(graph: &Graph) -> Vec<Vec<(usize, NodeInfo)>> {
    let mut layers: Vec<Vec<(usize, NodeInfo)>> = Vec::new();
    
    layers.push(graph.tensors.iter().enumerate()
        .filter(|(_, t)| !t.name.starts_with("tmp_"))
        .map(|(i, _)| (i, NodeInfo::Input(i)))
        .collect());
    
    let mut op_layers: Vec<Vec<usize>> = Vec::new();
    let mut remaining_ops: Vec<usize> = (0..graph.ops.len()).collect();
    
    while !remaining_ops.is_empty() {
        let mut current_layer = Vec::new();
        let mut i = 0;
        
        while i < remaining_ops.len() {
            let op_idx = remaining_ops[i];
            let op = &graph.ops[op_idx];
            
            let can_place = op.inputs.iter().all(|&input_idx| {
                if !graph.tensors[input_idx].name.starts_with("tmp_") {
                    return true;
                }
                
                let producing_op_idx = if input_idx >= (graph.tensors.len() - graph.ops.len()) {
                    input_idx - (graph.tensors.len() - graph.ops.len())
                } else {
                    return true;
                };
                op_layers.iter().any(|layer| layer.contains(&producing_op_idx))
            });
            
            if can_place {
                current_layer.push(op_idx);
                remaining_ops.remove(i);
            } else {
                i += 1;
            }
        }
        
        if !current_layer.is_empty() {
            op_layers.push(current_layer);
        } else if !remaining_ops.is_empty() {
            current_layer.push(remaining_ops.remove(0));
            op_layers.push(current_layer);
        }
    }
    
    for layer in op_layers {
        let op_layer: Vec<(usize, NodeInfo)> = layer.iter()
            .map(|&op_idx| (1000 + op_idx, NodeInfo::Operation(op_idx)))
            .collect();
        layers.push(op_layer);
    }
    
    layers.push(vec![(2000, NodeInfo::Output)]);
    
    layers
}

fn position_nodes_in_layers(layers: &[Vec<(usize, NodeInfo)>], _graph: &Graph) -> std::collections::HashMap<usize, (i32, i32, NodeInfo)> {
    let mut positioned = std::collections::HashMap::new();
    
    let canvas_width = 1080;
    let canvas_height = 1100;
    let margin_x = 120;
    let margin_y = 100;
    
    let usable_width = canvas_width - (2 * margin_x);
    let usable_height = canvas_height - (2 * margin_y);
    
    let layer_height = if layers.len() > 1 {
        usable_height / (layers.len() - 1) as i32
    } else {
        0
    };
    let base_y = margin_y;
    
    for (layer_idx, layer) in layers.iter().enumerate() {
        let y = base_y + (layer_idx as i32 * layer_height);
        
        for (pos_in_layer, &(node_id, ref node_info)) in layer.iter().enumerate() {
            let x = if layer.len() == 1 {
                canvas_width / 2
            } else if layer.len() == 2 {
                let spacing = usable_width / 3;
                margin_x + spacing + (pos_in_layer as i32 * spacing)
            } else {
                let min_spacing = 180;
                let ideal_spacing = usable_width / (layer.len() - 1) as i32;
                let actual_spacing = ideal_spacing.max(min_spacing);
                
                let total_width = actual_spacing * (layer.len() - 1) as i32;
                let start_x = if total_width > usable_width {
                    margin_x - (total_width - usable_width) / 2
                } else {
                    margin_x + (usable_width - total_width) / 2
                };
                
                start_x + (pos_in_layer as i32 * actual_spacing)
            };
            
            positioned.insert(node_id, (x, y, node_info.clone()));
        }
    }
    
    positioned
}

fn create_clean_edges(graph: &Graph) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    
    let mut tensor_producers: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (op_idx, _op) in graph.ops.iter().enumerate() {
        let output_tensor_idx = graph.tensors.len() - graph.ops.len() + op_idx;
        tensor_producers.insert(output_tensor_idx, op_idx);
    }

    for (op_idx, op) in graph.ops.iter().enumerate() {
        let op_node_id = 1000 + op_idx;
        
        for &input_idx in &op.inputs {
            let tensor = &graph.tensors[input_idx];
            
            if !tensor.name.starts_with("tmp_") {
                edges.push((input_idx, op_node_id));
            } else {
                if let Some(&producing_op_idx) = tensor_producers.get(&input_idx) {
                    let producing_op_id = 1000 + producing_op_idx;
                    edges.push((producing_op_id, op_node_id));
                }
            }
        }
    }
    
    for &output_tensor_idx in &graph.outputs {
        if let Some(&producing_op_idx) = tensor_producers.get(&output_tensor_idx) {
            let producing_op_id = 1000 + producing_op_idx;
            edges.push((producing_op_id, 2000));
        } else {
            if !graph.ops.is_empty() {
                let last_op_idx = graph.ops.len() - 1;
                let last_op_id = 1000 + last_op_idx;
                edges.push((last_op_id, 2000));
            }
        }
    }
    
    edges
}


#[allow(dead_code)]
fn draw_program_text<DB: plotters::prelude::DrawingBackend>(
    area: &plotters::prelude::DrawingArea<DB, plotters::coord::Shift>, 
    program_text: &str, 
    title: &str
) -> Result<(), Box<dyn std::error::Error>>
where 
    DB::ErrorType: 'static,
{
    let title_font = ("sans-serif", 16u32).into_font().color(&BLACK);
    area.draw(&Text::new(
        title,
        (20, 20),
        title_font
    ))?;
    
    let text_font = ("monospace", 12u32).into_font().color(&BLACK);
    let lines: Vec<&str> = program_text.lines().collect();
    
    for (i, line) in lines.iter().enumerate() {
        let y_pos = 45 + (i * 18) as i32;
        if y_pos < 280 {
            area.draw(&Text::new(
                *line,
                (20, y_pos),
                text_font.clone()
            ))?;
        }
    }
    
    Ok(())
}

pub fn generate_side_by_side_comparison(original: &Graph, optimized: &Graph, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let clean_filename = format!("results/{}_comparison.png", clean_name(name));
    let filename_for_backend = clean_filename.clone();
    
    let root = BitMapBackend::new(&filename_for_backend, (2400, 1200)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let (left_area, right_area) = root.split_horizontally(1200);
    
    left_area.fill(&WHITE)?;
    right_area.fill(&WHITE)?;
    
    root.draw(&plotters::element::PathElement::new(
        vec![(1200, 0), (1200, 1200)],
        plotters::style::BLACK.stroke_width(2)
    ))?;
    
    draw_graph_in_area(&left_area.margin(60, 60, 50, 60), original)?;
    draw_graph_in_area(&right_area.margin(60, 60, 50, 60), optimized)?;
    
    root.present()?;
    Ok(clean_filename)
}

fn draw_graph_in_area<DB: plotters::prelude::DrawingBackend>(area: &plotters::prelude::DrawingArea<DB, plotters::coord::Shift>, graph: &Graph) -> Result<(), Box<dyn std::error::Error>> 
where 
    DB::ErrorType: 'static,
{
    let (nodes, edges) = layout_graph(graph);
    
    for (from_id, to_id) in edges {
        let from_node = nodes.iter().find(|n| n.id == from_id);
        let to_node = nodes.iter().find(|n| n.id == to_id);
        if let (Some(from), Some(to)) = (from_node, to_node) {
            let dx = to.x - from.x;
            let dy = to.y - from.y;
            let len = ((dx * dx + dy * dy) as f64).sqrt();
            
            if len > 0.0 {
                let box_height = 28;
                
                let start_x = from.x;
                let start_y = from.y + box_height;
                
                let end_x = to.x;
                let end_y = to.y - box_height;
                
                let edge_color = RGBColor(60, 60, 60);
                area.draw(&PathElement::new(
                    vec![(start_x, start_y), (end_x, end_y)],
                    edge_color.stroke_width(2)
                ))?;
                
            }
        }
    }
    
    for node in nodes {
        let label = match node.node_type {
            NodeType::Operation(ref kind, cost) => {
                let op_name = match kind {
                    OpKind::Add => "ADD",
                    OpKind::Mul => "MUL",
                    OpKind::SumReduce => "SUM_REDUCE",
                    OpKind::MaxReduce => "MAX_REDUCE",
                    OpKind::Reshape => "RESHAPE",
                    OpKind::Broadcast => "BROADCAST",
                    OpKind::Exp => "EXP",
                    OpKind::Log => "LOG",
                    OpKind::Sin => "SIN",
                    OpKind::Sqrt => "SQRT",
                    OpKind::Recip => "RECIP",
                    OpKind::Mod => "MOD",
                    OpKind::LessThan => "LESS_THAN",
                    OpKind::Constant => "CONST",
                    OpKind::Input(ref name) => name,
                };
                format!("{} ({})", op_name, cost)
            },
            _ => {
                let shape_str = node.shape.iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("{} [{}]", node.name, shape_str)
            }
        };
        
        let box_width = 85;
        let box_height = 28;

        let (fill_color, border_color) = match node.node_type {
            NodeType::Input => (RGBColor(230, 245, 255), RGBColor(70, 130, 180)),    
            NodeType::Output => (RGBColor(240, 255, 240), RGBColor(34, 139, 34)),   
            NodeType::Operation(ref kind, _) => match kind {
                OpKind::Add | OpKind::Mul => (RGBColor(255, 245, 230), RGBColor(255, 140, 0)), 
                OpKind::SumReduce | OpKind::MaxReduce => (RGBColor(245, 230, 255), RGBColor(147, 112, 219)), 
                OpKind::Reshape | OpKind::Broadcast => (RGBColor(255, 255, 230), RGBColor(218, 165, 32)), 
                OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Sqrt | OpKind::Recip => (RGBColor(230, 255, 245), RGBColor(60, 179, 113)), 
                OpKind::Mod | OpKind::LessThan => (RGBColor(255, 235, 255), RGBColor(199, 21, 133)), 
                OpKind::Constant => (RGBColor(250, 240, 230), RGBColor(160, 82, 45)),  
                OpKind::Input(_) => (RGBColor(230, 245, 255), RGBColor(70, 130, 180)),   
            }
        };
        
        let shadow_offset = 3;
        area.draw(&Rectangle::new(
            [(node.x - box_width + shadow_offset, node.y - box_height + shadow_offset), 
             (node.x + box_width + shadow_offset, node.y + box_height + shadow_offset)], 
            RGBColor(0, 0, 0).mix(0.2).filled()
        ))?;
        
        area.draw(&Rectangle::new(
            [(node.x - box_width, node.y - box_height), 
             (node.x + box_width, node.y + box_height)], 
            fill_color.filled()
        ))?;
        
        area.draw(&Rectangle::new(
            [(node.x - box_width, node.y - box_height), 
             (node.x + box_width, node.y + box_height)], 
            border_color.stroke_width(2)
        ))?;
        
        let font_size = 15u32;
        let font_weight = match node.node_type {
            NodeType::Input | NodeType::Output => "sans-serif-bold",
            _ => "sans-serif"
        };
        let text_color = match node.node_type {
            NodeType::Input => RGBColor(25, 25, 112),      
            NodeType::Output => RGBColor(0, 100, 0),       
            _ => RGBColor(33, 33, 33)                      
        };
        
        let font = (font_weight, font_size).into_font().color(&text_color);
        
        let char_width = font_size as f32 * 0.42;
        let text_width = (label.len() as f32 * char_width) as i32;
        let text_height = font_size as i32;
        
        let text_x = node.x - text_width / 2;
        let text_y = node.y - text_height / 3;
        
        area.draw(&Text::new(
            label,
            (text_x, text_y),
            font
        ))?;
    }
    
    Ok(())
}

pub fn generate_graph_image(graph: &Graph, name: &str, suffix: &str) -> Result<String, Box<dyn std::error::Error>> {
    let clean_filename = format!("results/{}{}.png", clean_name(name), suffix);
    let (nodes, edges) = layout_graph(graph);
    let filename_for_backend = clean_filename.clone();
    let root = BitMapBackend::new(&filename_for_backend, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    for (from_id, to_id) in edges {
        let from_node = nodes.iter().find(|n| n.id == from_id);
        let to_node = nodes.iter().find(|n| n.id == to_id);
        if let (Some(from), Some(to)) = (from_node, to_node) {
            let dx = to.x - from.x;
            let dy = to.y - from.y;
            let len = ((dx * dx + dy * dy) as f64).sqrt();
            
            if len > 0.0 {
                let box_height = 20;
                
                let start_x = from.x;
                let start_y = from.y + box_height;
                
                let end_x = to.x;
                let end_y = to.y - box_height;

                let edge_color = RGBColor(60, 60, 60);
                root.draw(&PathElement::new(
                    vec![(start_x, start_y), (end_x, end_y)],
                    edge_color.stroke_width(2)
                ))?;
                
            }
        }
    }
    for node in nodes {
        let label = match node.node_type {
            NodeType::Operation(ref kind, cost) => {
                let op_name = match kind {
                    OpKind::Add => "ADD",
                    OpKind::Mul => "MUL",
                    OpKind::SumReduce => "SUM_REDUCE",
                    OpKind::MaxReduce => "MAX_REDUCE",
                    OpKind::Reshape => "RESHAPE",
                    OpKind::Broadcast => "BROADCAST",
                    OpKind::Exp => "EXP",
                    OpKind::Log => "LOG",
                    OpKind::Sin => "SIN",
                    OpKind::Sqrt => "SQRT",
                    OpKind::Recip => "RECIP",
                    OpKind::Mod => "MOD",
                    OpKind::LessThan => "LESS_THAN",
                    OpKind::Constant => "CONST",
                    OpKind::Input(ref name) => name,
                };
                format!("{} ({})", op_name, cost)
            },
            _ => {
                let shape_str = node.shape.iter()
                    .map(|&s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("{} [{}]", node.name, shape_str)
            }
        };
        let (fill_color, border_color) = match node.node_type {
            NodeType::Input => (RGBColor(230, 245, 255), RGBColor(70, 130, 180)),
            NodeType::Output => (RGBColor(240, 255, 240), RGBColor(34, 139, 34)),
            NodeType::Operation(ref kind, _) => match kind {
                OpKind::Add | OpKind::Mul => (RGBColor(255, 245, 230), RGBColor(255, 140, 0)),
                OpKind::SumReduce | OpKind::MaxReduce => (RGBColor(245, 230, 255), RGBColor(147, 112, 219)),
                OpKind::Reshape | OpKind::Broadcast => (RGBColor(255, 255, 230), RGBColor(218, 165, 32)),
                OpKind::Exp | OpKind::Log | OpKind::Sin | OpKind::Sqrt | OpKind::Recip => (RGBColor(230, 255, 245), RGBColor(60, 179, 113)),
                OpKind::Mod | OpKind::LessThan => (RGBColor(255, 235, 255), RGBColor(199, 21, 133)),
                OpKind::Constant => (RGBColor(250, 240, 230), RGBColor(160, 82, 45)),
                OpKind::Input(_) => (RGBColor(230, 245, 255), RGBColor(70, 130, 180)),
            }
        };
        
        let box_width = 75;
        let box_height = 20;
        
        let shadow_offset = 2;
        root.draw(&Rectangle::new(
            [(node.x - box_width + shadow_offset, node.y - box_height + shadow_offset), 
             (node.x + box_width + shadow_offset, node.y + box_height + shadow_offset)], 
            RGBColor(0, 0, 0).mix(0.2).filled()
        ))?;
        
        root.draw(&Rectangle::new(
            [(node.x - box_width, node.y - box_height), 
             (node.x + box_width, node.y + box_height)], 
            fill_color.filled()
        ))?;
        root.draw(&Rectangle::new(
            [(node.x - box_width, node.y - box_height), 
             (node.x + box_width, node.y + box_height)], 
            border_color.stroke_width(2)
        ))?;
        
        let font_size = 13u32;
        let text_color = match node.node_type {
            NodeType::Input => RGBColor(25, 25, 112),
            NodeType::Output => RGBColor(0, 100, 0),
            _ => RGBColor(33, 33, 33)
        };
        
        let char_width = font_size as f32 * 0.42;
        let text_width = (label.len() as f32 * char_width) as i32;
        let text_height = font_size as i32;
        
        let text_x = node.x - text_width / 2;
        let text_y = node.y - text_height / 3;
        
        root.draw(&Text::new(
            label,
            (text_x, text_y),
            ("sans-serif", font_size).into_font().color(&text_color)
        ))?;
    }
    root.present()?;
    Ok(clean_filename)
} 