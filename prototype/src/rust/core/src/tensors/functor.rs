// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

use crate::tensors::hyper_graph_tensor::{HyperGraphTensor, TensorNode};
use nalgebra::Vector3;
use parry3d::math::Point;
use parry3d::query::PointQuery;
use parry3d::shape::Cuboid;

/// Functor Trait for Tensor Operations
///
/// Under Category Theory, a Functor F maps objects from category C to D, and morphisms f: X -> Y
/// to F(f): F(X) -> F(D).
/// In Rust's affine type system, we implement this as an Endofunctor over the category
/// of HyperGraph state. Instead of allocating a new HyperGraph $H'$ for every map (which would
/// violate real-time memory constraints in robotics), we leverage `&mut` references guarantees,
/// creating an isomorphism between pure functional F(A) -> F(B) and linear memory mutation.
pub trait TensorMap {
    /// Applies an Endomorphism to all Material nodes within the functorial context.
    fn map_materials<F>(&mut self, f: F)
    where
        F: FnMut(&mut f32, &mut f32);
}

impl TensorMap for HyperGraphTensor {
    fn map_materials<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut f32, &mut f32),
    {
        let mut g = self.graph.write().unwrap();
        for node in g.node_weights_mut() {
            if let TensorNode::Material {
                ref mut yield_stress,
                ref mut viscosity,
                ..
            } = node
            {
                f(yield_stress, viscosity);
            }
        }
    }
}

/// A "Slicer" Functor
/// Maps Geometry -> Toolpath (TrajTensor)
pub struct SlicerFunctor;

impl SlicerFunctor {
    /// Bounding Box Slicer (Phase 1 Refinement)
    pub fn slice_aabb(tensor: &HyperGraphTensor, node_id: u32, layer_height: f32) -> Vec<[f32; 5]> {
        let mut path = Vec::new();

        // 1. Retrieve Geometry Node and its Bounds
        let g = tensor.graph.read().unwrap();
        let bounds = if let Some(TensorNode::Geometry { bounds, .. }) =
            g.node_weight(petgraph::prelude::NodeIndex::new(node_id as usize))
        {
            *bounds
        } else {
            return Vec::new();
        };

        let [min_x, min_y, min_z, max_x, max_y, max_z] = bounds;

        // 2. Physics: Create Parry3D Shape (Cuboid)
        let half_extents = Vector3::new(
            (max_x - min_x) / 2.0,
            (max_y - min_y) / 2.0,
            (max_z - min_z) / 2.0,
        );
        let cuboid = Cuboid::new(half_extents);
        let center = Point::new(
            min_x + half_extents.x,
            min_y + half_extents.y,
            min_z + half_extents.z,
        );

        // 3. Slicing Logic
        let z_layers = ((max_z - min_z) / layer_height).floor() as usize;

        for i in 0..z_layers {
            let z = min_z + (i as f32) * layer_height;
            let step_y = 0.1;
            let y_steps = ((max_y - min_y) / step_y).floor() as usize;

            for j in 0..y_steps {
                let y = min_y + (j as f32) * step_y;
                let p_start = Point::new(min_x, y, z);
                let local_start = p_start - center.coords;

                if cuboid.contains_local_point(&Point::from(local_start)) {
                    // Simple Zig-Zag
                    let print_speed = 50.0;
                    let e_val = 0.1;
                    let (start_pt, end_pt) = if j % 2 == 0 {
                        (
                            [min_x, y, z, 0.0, print_speed],
                            [max_x, y, z, e_val, print_speed],
                        )
                    } else {
                        (
                            [max_x, y, z, 0.0, print_speed],
                            [min_x, y, z, e_val, print_speed],
                        )
                    };
                    path.push(start_pt);
                    path.push(end_pt);
                }
            }
        }
        path
    }
}
