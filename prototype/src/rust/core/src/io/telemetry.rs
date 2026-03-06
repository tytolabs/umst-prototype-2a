// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! Live WebXR Telemetry Streamer
//! Broadcasts the HyperGraphTensor state continuously to frontend visualizers.

use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::broadcast;
use warp::ws::{Message, WebSocket};
use warp::Filter;

/// Telemetry Pipeline wrapping a Tokio broadcast channel
#[derive(Clone)]
pub struct TelemetryStreamer {
    sender: broadcast::Sender<String>,
}

impl TelemetryStreamer {
    /// Create a new streamer with a capacity of 16 buffered frames
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(16);
        TelemetryStreamer { sender }
    }

    /// Broadcast a HyperGraphTensor RenderState JSON object to all connected clients.
    /// This is lockless and completely decoupled from the PhysicsKernel thread.
    pub fn broadcast_state(&self, json_state: &str) {
        // We don't care if there are zero active listeners, so we ignore SendErrors.
        let _ = self.sender.send(json_state.to_string());
    }

    /// Mounts the telemetry WebSocket endpoint and the static HTML viewer on the provided warp router path
    pub fn spawn_server(streamer: Arc<Self>, port: u16) {
        let html_content = include_str!("../../static/telemetry_viewer.html").to_string();

        tokio::spawn(async move {
            let viewer_route =
                warp::path("viewer").map(move || warp::reply::html(html_content.clone()));

            let telemetry_route = warp::path("telemetry")
                .and(warp::ws())
                .and(warp::any().map(move || streamer.clone()))
                .map(|ws: warp::ws::Ws, state: Arc<TelemetryStreamer>| {
                    ws.on_upgrade(move |socket| handle_xr_client(socket, state))
                });

            let routes = viewer_route.or(telemetry_route).boxed();

            let addr: std::net::SocketAddr = ([0, 0, 0, 0], port).into();
            println!(
                "📡 WebXR Telemetry Stream mounted on ws://{}/telemetry",
                addr
            );
            println!("🌐 Digital Twin Viewer mounted on http://{}/viewer", addr);
            warp::serve(routes).run(addr).await;
        });
    }
}

async fn handle_xr_client(ws: WebSocket, streamer: Arc<TelemetryStreamer>) {
    let (mut client_ws_sender, _client_ws_rcv) = ws.split();
    let mut rx = streamer.sender.subscribe();

    println!("👁️  New XR Frontend client connected to live telemetry.");

    while let Ok(msg) = rx.recv().await {
        if let Err(e) = client_ws_sender.send(Message::text(msg)).await {
            eprintln!("XR Client disconnected: {}", e);
            break;
        }
    }
}
