// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy — Studio TYTO
//!
//! One-shot parity CLI: read one JSON proposal on stdin, print manifold + legacy gate JSON.

use std::io::{self, Read};

fn main() {
    let mut body = String::new();
    io::stdin()
        .read_to_string(&mut body)
        .expect("read stdin");
    let canonical = umst_core::manifold_gate_shim::evaluate_canonical_json(&body);
    let legacy = umst_core::manifold_gate_shim::evaluate_legacy_json(&body);
    println!("{{\"canonical\":{canonical},\"legacy\":{legacy}}}");
}
