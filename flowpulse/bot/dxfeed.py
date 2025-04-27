"""DXFeed WebSocket client implementation"""
import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class DXFeedStreamer:
    """DXFeed WebSocket client for streaming market data"""
    
    def __init__(self, token: str, url: str):
        """Initialize DXFeed streamer
        
        Args:
            token: API quote token from TastyTrade
            url: WebSocket URL for DXLink connection
        """
        self.token = token
        self.url = url
        self.ws = None
        self.running = False
        self.channels: Dict[int, Dict] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to DXLink websocket"""
        try:
            self.ws = await websockets.connect(self.url)
            self.running = True
            self.logger.info("Connected to DXLink websocket")
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat())
            
            # Start message handler
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            self.logger.error(f"Error connecting to DXLink: {e}")
            raise

    async def disconnect(self):
        """Disconnect from DXLink websocket"""
        if self.ws:
            await self.ws.close()
            self.running = False
            self.logger.info("Disconnected from DXLink")

    async def setup_channel(self, channel_id: int):
        """Setup a new channel
        
        Args:
            channel_id: Channel identifier
        """
        try:
            # Send SETUP message
            setup_msg = {
                "type": "SETUP",
                "channel": channel_id,
                "version": "0.1-DXF-JS/0.3.0",
                "keepaliveTimeout": 60,
                "acceptKeepaliveTimeout": 60
            }
            await self._send(setup_msg)
            self.channels[channel_id] = {"state": "setup"}
            
        except Exception as e:
            self.logger.error(f"Error setting up channel {channel_id}: {e}")
            raise

    async def authorize_channel(self, channel_id: int):
        """Authorize a channel using the API quote token
        
        Args:
            channel_id: Channel identifier
        """
        try:
            # Send AUTH message
            auth_msg = {
                "type": "AUTH",
                "channel": channel_id,
                "token": self.token
            }
            await self._send(auth_msg)
            
            # Wait for AUTH_STATE response
            response = await self._wait_for_message("AUTH_STATE", channel_id)
            if response.get("state") != "AUTHORIZED":
                raise Exception(f"Channel {channel_id} authorization failed")
                
            self.channels[channel_id]["state"] = "authorized"
            
        except Exception as e:
            self.logger.error(f"Error authorizing channel {channel_id}: {e}")
            raise

    async def subscribe(self, channel_id: int, symbol: str):
        """Subscribe to market data for a symbol
        
        Args:
            channel_id: Channel identifier
            symbol: Symbol to subscribe to
        """
        try:
            # Send FEED_SUBSCRIPTION message
            sub_msg = {
                "type": "FEED_SUBSCRIPTION",
                "channel": channel_id,
                "reset": True,
                "symbols": [symbol]
            }
            await self._send(sub_msg)
            
            if "symbols" not in self.channels[channel_id]:
                self.channels[channel_id]["symbols"] = set()
            self.channels[channel_id]["symbols"].add(symbol)
            
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol} on channel {channel_id}: {e}")
            raise

    async def _send(self, message: Dict):
        """Send a message to DXLink
        
        Args:
            message: Message to send
        """
        if not self.ws:
            raise Exception("Not connected to DXLink")
            
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending message to DXLink: {e}")
            raise

    async def _wait_for_message(self, msg_type: str, channel_id: int, timeout: int = 5) -> Dict:
        """Wait for a specific message type from DXLink
        
        Args:
            msg_type: Type of message to wait for
            channel_id: Channel identifier
            timeout: Timeout in seconds
            
        Returns:
            Message received
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for {msg_type} message")
                
            try:
                msg = json.loads(await self.ws.recv())
                if msg.get("type") == msg_type and msg.get("channel") == channel_id:
                    return msg
            except Exception as e:
                self.logger.error(f"Error waiting for message: {e}")
                raise

    async def _heartbeat(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                await self._send({"type": "KEEPALIVE", "channel": 0})
                await asyncio.sleep(30)
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                if not self.running:
                    break

    async def _handle_messages(self):
        """Handle incoming messages from DXLink"""
        while self.running:
            try:
                message = await self.ws.recv()
                msg = json.loads(message)
                
                # Handle different message types
                msg_type = msg.get("type")
                if msg_type == "FEED_DATA":
                    await self._handle_feed_data(msg)
                elif msg_type == "AUTH_STATE":
                    self.logger.info(f"Auth state update: {msg.get('state')}")
                elif msg_type == "ERROR":
                    self.logger.error(f"DXLink error: {msg}")
                    
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
                if not self.running:
                    break

    async def _handle_feed_data(self, message: Dict):
        """Handle FEED_DATA messages
        
        Args:
            message: FEED_DATA message
        """
        try:
            data = message.get("data", [])
            for item in data:
                symbol = item.get("symbol")
                if symbol and symbol in self.handlers:
                    for handler in self.handlers[symbol]:
                        await handler(item)
                        
        except Exception as e:
            self.logger.error(f"Error handling feed data: {e}")
