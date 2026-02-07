"""
WebSocket Router
Phase 6.1 - Core Infrastructure

Real-time communication for progress updates and notifications.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

router = APIRouter()

# Connection manager
class ConnectionManager:
    """Manages WebSocket connections and channel subscriptions."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # channel -> set of connection_ids

    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove from all subscriptions
        for channel in self.subscriptions.values():
            channel.discard(connection_id)

    def subscribe(self, connection_id: str, channel: str):
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        self.subscriptions[channel].add(connection_id)

    def unsubscribe(self, connection_id: str, channel: str):
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(connection_id)

    async def send_personal(self, connection_id: str, message: dict):
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json(message)

    async def broadcast_to_channel(self, channel: str, message: dict):
        if channel in self.subscriptions:
            for connection_id in self.subscriptions[channel]:
                if connection_id in self.active_connections:
                    try:
                        await self.active_connections[connection_id].send_json(message)
                    except Exception:
                        pass  # Connection might have closed


manager = ConnectionManager()


@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    import uuid
    connection_id = str(uuid.uuid4())

    await manager.connect(websocket, connection_id)

    # Send welcome message
    await manager.send_personal(connection_id, {
        "type": "connected",
        "connection_id": connection_id,
        "message": "Connected to CondorBrain WebSocket",
    })

    try:
        while True:
            # Receive and parse message
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            # Handle different message types
            msg_type = message.get("type")

            if msg_type == "subscribe":
                channel = message.get("channel")
                if channel:
                    manager.subscribe(connection_id, channel)
                    await manager.send_personal(connection_id, {
                        "type": "subscribed",
                        "channel": channel,
                    })

            elif msg_type == "unsubscribe":
                channel = message.get("channel")
                if channel:
                    manager.unsubscribe(connection_id, channel)
                    await manager.send_personal(connection_id, {
                        "type": "unsubscribed",
                        "channel": channel,
                    })

            elif msg_type == "ping":
                await manager.send_personal(connection_id, {
                    "type": "pong",
                    "timestamp": message.get("timestamp"),
                })

            else:
                await manager.send_personal(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        manager.disconnect(connection_id)


# Helper function for other modules to broadcast
async def broadcast_progress(channel: str, data: dict):
    """Broadcast progress update to all subscribers of a channel."""
    await manager.broadcast_to_channel(channel, {
        "type": "progress",
        "channel": channel,
        "data": data,
    })


# Export manager for use by other modules
def get_ws_manager() -> ConnectionManager:
    return manager
