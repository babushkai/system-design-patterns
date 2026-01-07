# WebSockets

## TL;DR

WebSocket is a full-duplex communication protocol that enables real-time, bidirectional data exchange over a single TCP connection. After an HTTP handshake upgrades the connection, both client and server can send messages independently. WebSockets are ideal for chat, gaming, collaborative editing, and any application requiring low-latency bidirectional communication.

---

## How WebSockets Work

```
HTTP Handshake (Upgrade):

Client                                        Server
  │                                              │
  │──── GET /chat HTTP/1.1 ─────────────────────►│
  │     Host: server.example.com                 │
  │     Upgrade: websocket                       │
  │     Connection: Upgrade                      │
  │     Sec-WebSocket-Key: dGhlIHNhbXBsZS...    │
  │     Sec-WebSocket-Version: 13                │
  │                                              │
  │◄─── HTTP/1.1 101 Switching Protocols ───────│
  │     Upgrade: websocket                       │
  │     Connection: Upgrade                      │
  │     Sec-WebSocket-Accept: s3pPLMBi...       │
  │                                              │
  │══════════ WebSocket Connection ══════════════│
  │                                              │
  │◄──── "Hello from server" ───────────────────│
  │                                              │
  │──── "Hello from client" ────────────────────►│
  │                                              │
  │◄──── "Real-time update" ────────────────────│
  │                                              │
  │──── "User action" ──────────────────────────►│
  │                                              │
```

---

## WebSocket Frame Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set to 1  |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+

Opcodes:
  0x0: Continuation frame
  0x1: Text frame
  0x2: Binary frame
  0x8: Connection close
  0x9: Ping
  0xA: Pong
```

---

## Basic Implementation

### Server-Side (Python with websockets library)

```python
import asyncio
import websockets
import json
from dataclasses import dataclass
from typing import Set, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Client:
    websocket: websockets.WebSocketServerProtocol
    user_id: str
    channels: Set[str]

class WebSocketServer:
    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.channels: Dict[str, Set[str]] = {}  # channel -> client_ids
    
    async def register(self, websocket, user_id: str) -> Client:
        """Register a new client connection."""
        client = Client(
            websocket=websocket,
            user_id=user_id,
            channels=set()
        )
        self.clients[user_id] = client
        logger.info(f"Client {user_id} connected")
        return client
    
    async def unregister(self, user_id: str):
        """Unregister client and clean up subscriptions."""
        if user_id in self.clients:
            client = self.clients[user_id]
            for channel in client.channels:
                if channel in self.channels:
                    self.channels[channel].discard(user_id)
            del self.clients[user_id]
            logger.info(f"Client {user_id} disconnected")
    
    async def subscribe(self, user_id: str, channel: str):
        """Subscribe client to channel."""
        if user_id in self.clients:
            self.clients[user_id].channels.add(channel)
            if channel not in self.channels:
                self.channels[channel] = set()
            self.channels[channel].add(user_id)
    
    async def unsubscribe(self, user_id: str, channel: str):
        """Unsubscribe client from channel."""
        if user_id in self.clients:
            self.clients[user_id].channels.discard(channel)
        if channel in self.channels:
            self.channels[channel].discard(user_id)
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user."""
        if user_id in self.clients:
            try:
                await self.clients[user_id].websocket.send(json.dumps(message))
            except websockets.ConnectionClosed:
                await self.unregister(user_id)
    
    async def broadcast_to_channel(self, channel: str, message: dict, exclude: str = None):
        """Broadcast message to all users in channel."""
        if channel in self.channels:
            for user_id in list(self.channels[channel]):
                if user_id != exclude:
                    await self.send_to_user(user_id, message)
    
    async def broadcast_all(self, message: dict):
        """Broadcast to all connected clients."""
        for user_id in list(self.clients.keys()):
            await self.send_to_user(user_id, message)
    
    async def handle_message(self, client: Client, raw_message: str):
        """Handle incoming message from client."""
        try:
            message = json.loads(raw_message)
            msg_type = message.get('type')
            
            if msg_type == 'subscribe':
                await self.subscribe(client.user_id, message['channel'])
                await self.send_to_user(client.user_id, {
                    'type': 'subscribed',
                    'channel': message['channel']
                })
            
            elif msg_type == 'unsubscribe':
                await self.unsubscribe(client.user_id, message['channel'])
            
            elif msg_type == 'message':
                # Broadcast message to channel
                await self.broadcast_to_channel(
                    message['channel'],
                    {
                        'type': 'message',
                        'channel': message['channel'],
                        'from': client.user_id,
                        'data': message['data']
                    }
                )
            
            elif msg_type == 'ping':
                await self.send_to_user(client.user_id, {'type': 'pong'})
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {client.user_id}")
    
    async def handler(self, websocket, path):
        """Main WebSocket connection handler."""
        # Extract user_id from query string or headers
        user_id = websocket.request_headers.get('X-User-ID', str(id(websocket)))
        
        client = await self.register(websocket, user_id)
        
        try:
            async for message in websocket:
                await self.handle_message(client, message)
        except websockets.ConnectionClosed:
            logger.info(f"Connection closed for {user_id}")
        finally:
            await self.unregister(user_id)

# Run server
server = WebSocketServer()

async def main():
    async with websockets.serve(server.handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

asyncio.run(main())
```

### Client-Side (JavaScript)

```javascript
class WebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = options;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectDelay = options.reconnectDelay || 1000;
    this.pingInterval = options.pingInterval || 30000;
    this.pingTimer = null;
    this.callbacks = new Map();
    this.messageHandlers = new Map();
    this.messageId = 0;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startPingInterval();
        this.emit('connected');
        resolve();
      };

      this.ws.onclose = (event) => {
        console.log(`WebSocket closed: ${event.code}`);
        this.stopPingInterval();
        this.emit('disconnected', event);
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  handleMessage(data) {
    try {
      const message = JSON.parse(data);
      
      // Check for response to request
      if (message.id && this.callbacks.has(message.id)) {
        const { resolve, reject } = this.callbacks.get(message.id);
        this.callbacks.delete(message.id);
        
        if (message.error) {
          reject(new Error(message.error));
        } else {
          resolve(message);
        }
        return;
      }

      // Emit message by type
      const handler = this.messageHandlers.get(message.type);
      if (handler) {
        handler(message);
      }

      this.emit('message', message);
    } catch (error) {
      console.error('Failed to parse message:', error);
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      throw new Error('WebSocket not connected');
    }
  }

  // Send message and wait for response
  request(data, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const id = ++this.messageId;
      data.id = id;

      const timer = setTimeout(() => {
        this.callbacks.delete(id);
        reject(new Error('Request timeout'));
      }, timeout);

      this.callbacks.set(id, {
        resolve: (response) => {
          clearTimeout(timer);
          resolve(response);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        }
      });

      this.send(data);
    });
  }

  subscribe(channel) {
    return this.request({ type: 'subscribe', channel });
  }

  unsubscribe(channel) {
    this.send({ type: 'unsubscribe', channel });
  }

  publish(channel, data) {
    this.send({ type: 'message', channel, data });
  }

  on(type, handler) {
    this.messageHandlers.set(type, handler);
  }

  emit(event, data) {
    const handler = this.messageHandlers.get(event);
    if (handler) handler(data);
  }

  startPingInterval() {
    this.pingTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, this.pingInterval);
  }

  stopPingInterval() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  close() {
    this.stopPingInterval();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Usage
const ws = new WebSocketClient('wss://api.example.com/ws');

ws.on('connected', () => {
  ws.subscribe('chat:room1');
});

ws.on('message', (msg) => {
  if (msg.type === 'message') {
    displayMessage(msg.from, msg.data);
  }
});

ws.connect();
```

---

## Scaling WebSockets

### Redis Pub/Sub for Horizontal Scaling

```python
import asyncio
import aioredis
import json
from typing import Dict, Set

class ScalableWebSocketServer:
    """
    WebSocket server that scales horizontally using Redis pub/sub.
    Each server instance handles its own connections but broadcasts
    messages through Redis to reach all clients.
    """
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.redis_url = redis_url
        self.redis = None
        self.pubsub = None
        
        # Local connections only
        self.local_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.local_subscriptions: Dict[str, Set[str]] = {}  # channel -> user_ids
    
    async def connect_redis(self):
        """Initialize Redis connection."""
        self.redis = await aioredis.from_url(self.redis_url)
        self.pubsub = self.redis.pubsub()
        
        # Start listener for Redis messages
        asyncio.create_task(self._redis_listener())
    
    async def _redis_listener(self):
        """Listen for messages from Redis and deliver to local clients."""
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode()
                data = json.loads(message['data'])
                
                # Deliver to local subscribers only
                await self._deliver_locally(channel, data)
    
    async def _deliver_locally(self, channel: str, message: dict):
        """Deliver message to local clients subscribed to channel."""
        local_subscribers = self.local_subscriptions.get(channel, set())
        
        for user_id in local_subscribers:
            if user_id in self.local_clients:
                try:
                    await self.local_clients[user_id].send(json.dumps(message))
                except:
                    pass
    
    async def subscribe(self, user_id: str, channel: str):
        """Subscribe user to channel."""
        # Track locally
        if channel not in self.local_subscriptions:
            self.local_subscriptions[channel] = set()
            # Subscribe to Redis channel
            await self.pubsub.subscribe(channel)
        
        self.local_subscriptions[channel].add(user_id)
    
    async def publish(self, channel: str, message: dict):
        """Publish message to channel (all server instances)."""
        # Publish through Redis
        await self.redis.publish(channel, json.dumps(message))
    
    async def register(self, websocket, user_id: str):
        """Register local connection."""
        self.local_clients[user_id] = websocket
        
        # Store connection info in Redis for presence
        await self.redis.hset(
            'ws:connections',
            user_id,
            json.dumps({
                'server': self.server_id,
                'connected_at': time.time()
            })
        )
    
    async def unregister(self, user_id: str):
        """Unregister connection."""
        if user_id in self.local_clients:
            del self.local_clients[user_id]
        
        # Remove from all local subscriptions
        for subscribers in self.local_subscriptions.values():
            subscribers.discard(user_id)
        
        # Remove from Redis
        await self.redis.hdel('ws:connections', user_id)
```

```
Horizontal Scaling Architecture:

                    ┌─────────────────────────────────────┐
                    │           Load Balancer             │
                    │     (WebSocket aware, sticky)       │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
    ┌───────────┐               ┌───────────┐               ┌───────────┐
    │  Server 1 │               │  Server 2 │               │  Server 3 │
    │           │               │           │               │           │
    │ Clients:  │               │ Clients:  │               │ Clients:  │
    │ [A, B, C] │               │ [D, E]    │               │ [F, G, H] │
    └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │      Redis      │
                            │    Pub/Sub      │
                            └─────────────────┘

User A sends message to channel "room1":
1. Server 1 receives WebSocket message from A
2. Server 1 publishes to Redis channel "room1"
3. All servers receive from Redis
4. Each server delivers to its local clients subscribed to "room1"
```

### Connection State with Redis

```python
class ConnectionState:
    """Manage WebSocket connection state in Redis."""
    
    def __init__(self, redis, server_id: str):
        self.redis = redis
        self.server_id = server_id
        self.connection_ttl = 300  # 5 minutes
    
    async def set_connected(self, user_id: str, metadata: dict = None):
        """Mark user as connected."""
        data = {
            'server': self.server_id,
            'connected_at': time.time(),
            **(metadata or {})
        }
        
        pipeline = self.redis.pipeline()
        pipeline.hset('ws:connections', user_id, json.dumps(data))
        pipeline.sadd(f'ws:server:{self.server_id}', user_id)
        pipeline.setex(f'ws:heartbeat:{user_id}', self.connection_ttl, '1')
        await pipeline.execute()
    
    async def heartbeat(self, user_id: str):
        """Update connection heartbeat."""
        await self.redis.setex(f'ws:heartbeat:{user_id}', self.connection_ttl, '1')
    
    async def set_disconnected(self, user_id: str):
        """Mark user as disconnected."""
        pipeline = self.redis.pipeline()
        pipeline.hdel('ws:connections', user_id)
        pipeline.srem(f'ws:server:{self.server_id}', user_id)
        pipeline.delete(f'ws:heartbeat:{user_id}')
        await pipeline.execute()
    
    async def is_connected(self, user_id: str) -> bool:
        """Check if user is connected (any server)."""
        return await self.redis.exists(f'ws:heartbeat:{user_id}')
    
    async def get_connection(self, user_id: str) -> dict:
        """Get user's connection info."""
        data = await self.redis.hget('ws:connections', user_id)
        return json.loads(data) if data else None
    
    async def get_server_connections(self) -> list:
        """Get all connections on this server."""
        return await self.redis.smembers(f'ws:server:{self.server_id}')
    
    async def cleanup_stale(self):
        """Clean up stale connections (heartbeat expired)."""
        connections = await self.redis.smembers(f'ws:server:{self.server_id}')
        
        for user_id in connections:
            user_id = user_id.decode() if isinstance(user_id, bytes) else user_id
            if not await self.is_connected(user_id):
                await self.set_disconnected(user_id)
```

---

## Message Protocol Design

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any
import json

class MessageType(Enum):
    # Control messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    
    # Pub/Sub
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PUBLISH = "publish"
    MESSAGE = "message"
    
    # Request/Response
    REQUEST = "request"
    RESPONSE = "response"

@dataclass
class Message:
    type: MessageType
    id: Optional[str] = None
    channel: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.type.value,
            'id': self.id,
            'channel': self.channel,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp or time.time()
        })
    
    @classmethod
    def from_json(cls, raw: str) -> 'Message':
        data = json.loads(raw)
        return cls(
            type=MessageType(data['type']),
            id=data.get('id'),
            channel=data.get('channel'),
            data=data.get('data'),
            error=data.get('error'),
            timestamp=data.get('timestamp')
        )

class MessageHandler:
    """Route messages to handlers based on type."""
    
    def __init__(self):
        self.handlers = {}
    
    def register(self, msg_type: MessageType):
        def decorator(func):
            self.handlers[msg_type] = func
            return func
        return decorator
    
    async def handle(self, client, message: Message):
        handler = self.handlers.get(message.type)
        if handler:
            return await handler(client, message)
        else:
            return Message(
                type=MessageType.ERROR,
                id=message.id,
                error=f'Unknown message type: {message.type.value}'
            )

# Usage
handler = MessageHandler()

@handler.register(MessageType.SUBSCRIBE)
async def handle_subscribe(client, message: Message):
    await server.subscribe(client.user_id, message.channel)
    return Message(
        type=MessageType.RESPONSE,
        id=message.id,
        data={'subscribed': message.channel}
    )

@handler.register(MessageType.PUBLISH)
async def handle_publish(client, message: Message):
    await server.broadcast_to_channel(
        message.channel,
        Message(
            type=MessageType.MESSAGE,
            channel=message.channel,
            data=message.data
        ),
        exclude=client.user_id
    )
    return Message(
        type=MessageType.RESPONSE,
        id=message.id,
        data={'published': True}
    )
```

---

## Authentication and Security

```python
import jwt
from functools import wraps

class WebSocketAuth:
    """WebSocket authentication middleware."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    async def authenticate(self, websocket) -> dict:
        """Authenticate WebSocket connection."""
        # Method 1: Token in query string
        token = websocket.query_params.get('token')
        
        # Method 2: Token in Sec-WebSocket-Protocol header
        if not token:
            protocols = websocket.request_headers.get('Sec-WebSocket-Protocol', '')
            for protocol in protocols.split(','):
                if protocol.strip().startswith('auth.'):
                    token = protocol.strip()[5:]
                    break
        
        # Method 3: Send token as first message
        if not token:
            try:
                first_message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=5.0
                )
                auth_data = json.loads(first_message)
                if auth_data.get('type') == 'auth':
                    token = auth_data.get('token')
            except asyncio.TimeoutError:
                raise AuthenticationError("Authentication timeout")
        
        if not token:
            raise AuthenticationError("No authentication token provided")
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

class AuthenticatedWebSocketServer(WebSocketServer):
    """WebSocket server with authentication."""
    
    def __init__(self, auth: WebSocketAuth):
        super().__init__()
        self.auth = auth
    
    async def handler(self, websocket, path):
        try:
            # Authenticate first
            user_info = await self.auth.authenticate(websocket)
            user_id = user_info['user_id']
            
            # Send auth success
            await websocket.send(json.dumps({
                'type': 'authenticated',
                'user_id': user_id
            }))
            
            # Continue with normal handling
            client = await self.register(websocket, user_id)
            client.user_info = user_info
            
            async for message in websocket:
                await self.handle_message(client, message)
                
        except AuthenticationError as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': str(e)
            }))
            await websocket.close(4001, 'Authentication failed')
        finally:
            if 'client' in locals():
                await self.unregister(user_id)
```

---

## Rate Limiting

```python
import time
from collections import defaultdict

class WebSocketRateLimiter:
    """Rate limit WebSocket messages per client."""
    
    def __init__(
        self,
        messages_per_second: int = 10,
        burst_size: int = 20,
        disconnect_on_exceed: bool = False
    ):
        self.rate = messages_per_second
        self.burst = burst_size
        self.disconnect_on_exceed = disconnect_on_exceed
        self.tokens = defaultdict(lambda: burst_size)
        self.last_update = defaultdict(time.time)
    
    def check_rate(self, user_id: str) -> tuple[bool, str]:
        """Check if message is allowed. Returns (allowed, reason)."""
        now = time.time()
        
        # Refill tokens
        elapsed = now - self.last_update[user_id]
        self.tokens[user_id] = min(
            self.burst,
            self.tokens[user_id] + elapsed * self.rate
        )
        self.last_update[user_id] = now
        
        if self.tokens[user_id] >= 1:
            self.tokens[user_id] -= 1
            return True, ""
        else:
            return False, "Rate limit exceeded"
    
    def reset(self, user_id: str):
        """Reset rate limit for user."""
        self.tokens[user_id] = self.burst
        self.last_update[user_id] = time.time()

# Integration
rate_limiter = WebSocketRateLimiter(messages_per_second=10)

async def handle_message(self, client, message):
    allowed, reason = rate_limiter.check_rate(client.user_id)
    
    if not allowed:
        await self.send_to_user(client.user_id, {
            'type': 'error',
            'error': reason
        })
        
        if rate_limiter.disconnect_on_exceed:
            await client.websocket.close(4008, 'Rate limit exceeded')
        return
    
    # Process message normally
    await self._process_message(client, message)
```

---

## Load Balancer Configuration

### nginx (WebSocket support)

```nginx
upstream websocket_backend {
    # Sticky sessions required for WebSocket
    ip_hash;
    
    server backend1:8765;
    server backend2:8765;
    server backend3:8765;
}

server {
    listen 443 ssl;
    
    location /ws {
        proxy_pass http://websocket_backend;
        
        # WebSocket upgrade
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        
        # Disable buffering
        proxy_buffering off;
    }
}
```

---

## Key Takeaways

1. **Bidirectional communication**: WebSocket enables real-time two-way messaging over a single connection

2. **Proper handshake**: Connection starts with HTTP upgrade; handle authentication before or during handshake

3. **Connection management**: Track connections, handle disconnects gracefully, implement heartbeat/ping-pong

4. **Horizontal scaling**: Use Redis pub/sub or similar to broadcast messages across server instances

5. **Rate limiting**: Protect against message flooding with per-client rate limits

6. **Reconnection logic**: Clients should implement exponential backoff reconnection

7. **Load balancer configuration**: Requires sticky sessions and WebSocket-aware configuration
