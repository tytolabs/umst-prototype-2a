# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
# SPDX-License-Identifier: MIT

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random

class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # We accept the POST request from the Rust llm_client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # We intentionally return aggressive/unsafe values sometimes to trigger the DUMSTO Vetoes
        unsafe_torque = random.uniform(-5.0, 20.0) 
        unsafe_flow = random.uniform(0.0, 5.0)
        
        mock_completion = {
            "id": "chatcmpl-mock",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "reasoning_trace": [
                            "I calculate the volumetric pressure.",
                            "I believe this torque is safe according to the extruder specs.",
                            "Focusing on speed."
                        ],
                        "intended_pyramid_layer": random.choice(["L1", "L3"]),
                        "delta_torque_nm": unsafe_torque,
                        "delta_flow_rate_lpm": unsafe_flow,
                        "confidence": 0.88
                    })
                }
            }]
        }
        
        self.wfile.write(json.dumps(mock_completion).encode('utf-8'))
        
    def log_message(self, format, *args):
        # Suppress logging to keep the terminal clean
        pass

if __name__ == '__main__':
    port = 11434
    server = HTTPServer(('localhost', port), MockOpenAIHandler)
    print(f"Mock OpenAI endpoint running on http://localhost:{port}/v1/chat/completions")
    server.serve_forever()
