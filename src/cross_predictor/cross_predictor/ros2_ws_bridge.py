"""
ros2_ws_bridge.py
=================
Bridges your ROS2 prediction topic to a WebSocket server so the
browser dashboard can consume it in real time.

Architecture:
    ROS2 topic (/pedestrian_predictions)
          ↓
    This node (rclpy subscriber)
          ↓
    WebSocket server (ws://localhost:8765)
          ↓
    Browser dashboard

Requirements:
    pip install websockets
    ROS2 (Humble / Iron / Jazzy) with rclpy
    Your custom message OR std_msgs/String with JSON payload

Usage:
    # Terminal 1 — run your predictor pipeline as normal
    # Terminal 2 — run this bridge
    python3 ros2_ws_bridge.py --topic /pedestrian_predictions --port 8765

    # If your message is a custom type, set --msg-type accordingly.
    # Default assumes std_msgs/String with a JSON string payload.

Notes:
    - The bridge is read-only; it never publishes back to ROS2.
    - Multiple browser clients can connect simultaneously.
    - Messages are forwarded as-is (JSON string) to all connected clients.
"""

import argparse
import asyncio
import json
import threading
import ast
import re
import logging
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ros2_ws_bridge")

# ─────────────────────────────────────────────────────────────────────────────
# Shared state between ROS2 thread and asyncio thread
# ─────────────────────────────────────────────────────────────────────────────
connected_clients: set = set()
message_queue: asyncio.Queue = None          # created inside asyncio loop
_loop: asyncio.AbstractEventLoop = None


def enqueue_message(payload: dict) -> None:
    """Thread-safe enqueue called from the ROS2 thread."""
    if _loop and message_queue:
        _loop.call_soon_threadsafe(message_queue.put_nowait, payload)


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 subscriber node
# ─────────────────────────────────────────────────────────────────────────────
class PredictionBridgeNode(Node):
    def __init__(self, topic: str):
        super().__init__("pedestrian_ws_bridge")
        self.sub = self.create_subscription(
            String,
            topic,
            self._on_message,
            qos_profile=10,
        )
        log.info(f"Subscribed to ROS2 topic: {topic}")

    def _on_message(self, msg: String):
        log.info(f"Received message on ROS2 topic: {msg.data[:100]}...")  # Log first 100 chars
        raw = msg.data.strip()
        try:
            # Try JSON first, then Python dict literal (ast.literal_eval)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = ast.literal_eval(raw)

            payload["_bridge_ts"] = datetime.utcnow().isoformat() + "Z"
            enqueue_message(payload)

        except Exception as e:
            log.warning(f"Could not parse message: {e}  raw={raw[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket server
# ─────────────────────────────────────────────────────────────────────────────
async def ws_handler(websocket) -> None:
    connected_clients.add(websocket)
    log.info(f"Client connected: {websocket.remote_address}  total={len(connected_clients)}")
    try:
        # Send a handshake so the dashboard knows it's live
        await websocket.send(json.dumps({"type": "connected", "msg": "ROS2 bridge ready"}))
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        log.info(f"Client disconnected  total={len(connected_clients)}")


async def broadcast_loop() -> None:
    """Read from queue and fan-out to all connected WebSocket clients."""
    while True:
        payload = await message_queue.get()
        if not connected_clients:
            continue
        data = json.dumps(payload)
        # Fire and forget to all clients; remove stale ones
        stale = set()
        for ws in list(connected_clients):
            try:
                await ws.send(data)
            except websockets.ConnectionClosed:
                stale.add(ws)
        connected_clients.difference_update(stale)


async def run_ws_server(host: str, port: int) -> None:
    global message_queue, _loop
    _loop = asyncio.get_running_loop()
    message_queue = asyncio.Queue(maxsize=500)

    async with websockets.serve(ws_handler, host, port):
        log.info(f"WebSocket server listening on ws://{host}:{port}")
        await broadcast_loop()          # runs forever


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — ROS2 spin in a thread, asyncio in main thread
# ─────────────────────────────────────────────────────────────────────────────
def ros2_spin_thread(node) -> None:
    """Spin an already-created node. rclpy.init() must have been called before."""
    try:
        rclpy.spin(node)
    except Exception as e:
        log.error(f"ROS2 spin error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    # Init once — ros2_spin_thread reuses this context, never calls init() again
    rclpy.init()

    # Read ROS2 parameters (works with ros2 run and launch files)
    param_node = rclpy.create_node('ws_bridge_params')
    param_node.declare_parameter('topic', '/cross_predictor/final')
    param_node.declare_parameter('host',  '0.0.0.0')
    param_node.declare_parameter('port',  8765)

    topic = param_node.get_parameter('topic').value
    host  = param_node.get_parameter('host').value
    port  = param_node.get_parameter('port').value
    param_node.destroy_node()

    # Create the subscriber node, spin it in a background thread
    node = PredictionBridgeNode(topic)
    t = threading.Thread(target=ros2_spin_thread, args=(node,), daemon=True)
    t.start()

    try:
        asyncio.run(run_ws_server(host, port))
    except KeyboardInterrupt:
        log.info("Bridge stopped.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()