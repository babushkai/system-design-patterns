# Server-Sent Events (SSE)

## TL;DR

Server-Sent Events (SSE) is a simple, HTTP-based protocol for streaming updates from server to client. Unlike WebSockets, SSE is unidirectional (server to client only), uses standard HTTP, and includes automatic reconnection. It's ideal for dashboards, notifications, feeds, and any scenario where the client only needs to receive updates.

---

## How SSE Works

```
Client                                        Server
  │                                              │
  │──── GET /events ────────────────────────────►│
  │     Accept: text/event-stream                │
  │                                              │
  │◄──── HTTP 200 ───────────────────────────────│
  │      Content-Type: text/event-stream         │
  │      (connection stays open)                 │
  │                                              │
  │◄──── data: {"price": 150.00}\n\n ───────────│
  │                                              │
  │◄──── data: {"price": 151.25}\n\n ───────────│
  │                                              │
  │◄──── data: {"price": 149.50}\n\n ───────────│
  │                                              │
  │              ... (continuous) ...            │
  │                                              │
  │      (connection drops)                      │
  │                                              │
  │──── GET /events ────────────────────────────►│
  │     Last-Event-ID: 42                        │
  │     (automatic reconnection!)                │
```

---

## SSE Message Format

```
Single message:
data: Hello World\n
\n

Multi-line message:
data: first line\n
data: second line\n
data: third line\n
\n

With event type:
event: notification\n
data: {"message": "New follower"}\n
\n

With ID (for reconnection):
id: 42\n
event: update\n
data: {"value": 100}\n
\n

Setting retry interval:
retry: 5000\n

Comment (keepalive):
: this is a comment\n
```

---

## Basic Implementation

### Server-Side (Python/Flask)

```python
from flask import Flask, Response, request
import json
import time
from typing import Generator
import queue
import threading

app = Flask(__name__)

class SSEManager:
    """Manage SSE connections and broadcasting."""
    
    def __init__(self):
        self.clients: dict[str, queue.Queue] = {}
        self.message_id = 0
        self.lock = threading.Lock()
    
    def register(self, client_id: str) -> queue.Queue:
        """Register a new client."""
        client_queue = queue.Queue()
        with self.lock:
            self.clients[client_id] = client_queue
        return client_queue
    
    def unregister(self, client_id: str):
        """Unregister a client."""
        with self.lock:
            self.clients.pop(client_id, None)
    
    def broadcast(self, data: dict, event: str = None):
        """Send message to all clients."""
        with self.lock:
            self.message_id += 1
            msg_id = self.message_id
            clients = list(self.clients.values())
        
        for client_queue in clients:
            try:
                client_queue.put_nowait({
                    'id': msg_id,
                    'event': event,
                    'data': data
                })
            except queue.Full:
                pass  # Client too slow, skip
    
    def send_to(self, client_id: str, data: dict, event: str = None):
        """Send message to specific client."""
        with self.lock:
            if client_id in self.clients:
                self.message_id += 1
                try:
                    self.clients[client_id].put_nowait({
                        'id': self.message_id,
                        'event': event,
                        'data': data
                    })
                except queue.Full:
                    pass

sse_manager = SSEManager()

def format_sse(data: dict, event: str = None, id: int = None) -> str:
    """Format data as SSE message."""
    lines = []
    
    if id is not None:
        lines.append(f'id: {id}')
    
    if event:
        lines.append(f'event: {event}')
    
    # Handle multi-line data
    json_data = json.dumps(data)
    lines.append(f'data: {json_data}')
    
    return '\n'.join(lines) + '\n\n'

def event_stream(client_id: str, last_event_id: int = None) -> Generator:
    """Generate SSE stream for client."""
    client_queue = sse_manager.register(client_id)
    
    try:
        # Send any missed messages if reconnecting
        if last_event_id:
            missed = get_messages_since(last_event_id)
            for msg in missed:
                yield format_sse(msg['data'], msg.get('event'), msg['id'])
        
        # Send keepalive comment
        yield ': connected\n\n'
        
        while True:
            try:
                # Wait for message with timeout (for keepalive)
                message = client_queue.get(timeout=30)
                yield format_sse(
                    message['data'],
                    message.get('event'),
                    message.get('id')
                )
            except queue.Empty:
                # Send keepalive comment
                yield ': keepalive\n\n'
    finally:
        sse_manager.unregister(client_id)

@app.route('/events')
def sse_endpoint():
    """SSE endpoint."""
    client_id = request.args.get('client_id', request.remote_addr)
    last_event_id = request.headers.get('Last-Event-ID', type=int)
    
    response = Response(
        event_stream(client_id, last_event_id),
        mimetype='text/event-stream'
    )
    
    # Important headers
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    
    return response

@app.route('/publish', methods=['POST'])
def publish():
    """Publish event to all clients."""
    data = request.json
    event_type = data.pop('_event', None)
    sse_manager.broadcast(data, event_type)
    return {'status': 'published'}
```

### Client-Side (JavaScript)

```javascript
class SSEClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = options;
    this.eventSource = null;
    this.callbacks = {};
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectDelay = options.reconnectDelay || 1000;
  }

  connect() {
    // EventSource handles reconnection automatically
    this.eventSource = new EventSource(this.url);

    // Default message handler (no event type)
    this.eventSource.onmessage = (event) => {
      this.handleEvent('message', event);
    };

    // Connection opened
    this.eventSource.onopen = () => {
      console.log('SSE connected');
      this.reconnectAttempts = 0;
      this.emit('connected');
    };

    // Error handling
    this.eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      
      if (this.eventSource.readyState === EventSource.CLOSED) {
        this.emit('disconnected');
        this.handleReconnect();
      }
    };

    return this;
  }

  // Listen for specific event types
  on(eventType, callback) {
    if (!this.callbacks[eventType]) {
      this.callbacks[eventType] = [];
      
      // Register with EventSource for custom event types
      if (this.eventSource && eventType !== 'message' && 
          eventType !== 'connected' && eventType !== 'disconnected') {
        this.eventSource.addEventListener(eventType, (event) => {
          this.handleEvent(eventType, event);
        });
      }
    }
    
    this.callbacks[eventType].push(callback);
    return this;
  }

  handleEvent(eventType, event) {
    const data = JSON.parse(event.data);
    this.emit(eventType, data, event.lastEventId);
  }

  emit(eventType, data, id) {
    const callbacks = this.callbacks[eventType] || [];
    callbacks.forEach(cb => cb(data, id));
  }

  handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  close() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
}

// Usage
const sse = new SSEClient('/events?client_id=user123');

sse.on('connected', () => {
  console.log('Connected to event stream');
});

sse.on('message', (data) => {
  console.log('Received:', data);
});

// Custom event types
sse.on('notification', (data) => {
  showNotification(data);
});

sse.on('price-update', (data) => {
  updatePriceDisplay(data);
});

sse.connect();
```

---

## Scaling SSE with Redis

```python
import redis
import json
import threading
from typing import Dict, Set
import gevent
from gevent import queue as gqueue

class RedisSSEManager:
    """
    Scalable SSE using Redis pub/sub.
    Works across multiple server instances.
    """
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.redis = redis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
        self.local_clients: Dict[str, Dict[str, gqueue.Queue]] = {}
        self.subscriptions: Set[str] = set()
        self.lock = threading.Lock()
        
        # Message ID counter in Redis for consistency
        self.id_key = 'sse:message_id'
        
        # Start listener
        self.listener = gevent.spawn(self._listen)
    
    def _listen(self):
        """Listen to Redis pub/sub."""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode()
                data = json.loads(message['data'])
                self._distribute_locally(channel, data)
    
    def _distribute_locally(self, channel: str, data: dict):
        """Send message to local clients subscribed to channel."""
        with self.lock:
            clients = list(self.local_clients.get(channel, {}).values())
        
        for client_queue in clients:
            try:
                client_queue.put_nowait(data)
            except:
                pass
    
    def subscribe(self, channel: str, client_id: str) -> gqueue.Queue:
        """Subscribe client to channel."""
        client_queue = gqueue.Queue(maxsize=100)
        
        with self.lock:
            if channel not in self.local_clients:
                self.local_clients[channel] = {}
                self.pubsub.subscribe(channel)
                self.subscriptions.add(channel)
            
            self.local_clients[channel][client_id] = client_queue
        
        return client_queue
    
    def unsubscribe(self, channel: str, client_id: str):
        """Unsubscribe client from channel."""
        with self.lock:
            if channel in self.local_clients:
                self.local_clients[channel].pop(client_id, None)
                
                if not self.local_clients[channel]:
                    del self.local_clients[channel]
                    self.pubsub.unsubscribe(channel)
                    self.subscriptions.discard(channel)
    
    def publish(self, channel: str, data: dict, event: str = None) -> int:
        """Publish message to channel (all instances)."""
        message_id = self.redis.incr(self.id_key)
        
        message = {
            'id': message_id,
            'event': event,
            'data': data,
            'timestamp': time.time()
        }
        
        # Store in Redis for reconnection support
        self._store_message(channel, message)
        
        # Publish to all instances
        self.redis.publish(channel, json.dumps(message))
        
        return message_id
    
    def _store_message(self, channel: str, message: dict, ttl: int = 300):
        """Store message for reconnection support."""
        key = f'sse:history:{channel}'
        self.redis.zadd(key, {json.dumps(message): message['id']})
        self.redis.expire(key, ttl)
        
        # Trim to last 1000 messages
        self.redis.zremrangebyrank(key, 0, -1001)
    
    def get_messages_since(self, channel: str, last_id: int) -> list:
        """Get messages since last_id for reconnection."""
        key = f'sse:history:{channel}'
        messages = self.redis.zrangebyscore(
            key, 
            f'({last_id}',  # exclusive
            '+inf'
        )
        return [json.loads(m) for m in messages]

# Usage with Flask
redis_sse = RedisSSEManager()

def channel_stream(channel: str, client_id: str, last_id: int = None):
    """Generate SSE stream for a channel."""
    client_queue = redis_sse.subscribe(channel, client_id)
    
    try:
        # Replay missed messages
        if last_id:
            for msg in redis_sse.get_messages_since(channel, last_id):
                yield format_sse(msg['data'], msg.get('event'), msg['id'])
        
        yield ': connected\n\n'
        
        while True:
            try:
                message = client_queue.get(timeout=30)
                yield format_sse(
                    message['data'],
                    message.get('event'),
                    message['id']
                )
            except gqueue.Empty:
                yield ': keepalive\n\n'
    finally:
        redis_sse.unsubscribe(channel, client_id)

@app.route('/events/<channel>')
def channel_events(channel):
    client_id = request.args.get('client_id', str(uuid.uuid4()))
    last_id = request.headers.get('Last-Event-ID', type=int)
    
    return Response(
        channel_stream(channel, client_id, last_id),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )
```

---

## Common Use Cases

### Real-Time Dashboard

```python
import psutil
import time
from threading import Thread

class SystemMetricsPublisher:
    """Publish system metrics via SSE."""
    
    def __init__(self, sse_manager, interval: float = 1.0):
        self.sse = sse_manager
        self.interval = interval
        self.running = False
    
    def start(self):
        self.running = True
        Thread(target=self._publish_loop, daemon=True).start()
    
    def stop(self):
        self.running = False
    
    def _publish_loop(self):
        while self.running:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                },
                'timestamp': time.time()
            }
            
            self.sse.publish('metrics', metrics, event='system-metrics')
            time.sleep(self.interval)

# Start publishing
publisher = SystemMetricsPublisher(redis_sse)
publisher.start()
```

```javascript
// Dashboard client
const dashboard = new SSEClient('/events/metrics');

dashboard.on('system-metrics', (data) => {
  document.getElementById('cpu').textContent = `${data.cpu_percent}%`;
  document.getElementById('memory').textContent = `${data.memory_percent}%`;
  
  updateChart('cpu-chart', data.timestamp, data.cpu_percent);
  updateChart('memory-chart', data.timestamp, data.memory_percent);
});

dashboard.connect();
```

### Live Activity Feed

```python
class ActivityFeedPublisher:
    """Publish user activity events."""
    
    def __init__(self, sse_manager):
        self.sse = sse_manager
    
    def publish_activity(self, user_id: str, action: str, details: dict):
        """Publish activity event."""
        event_data = {
            'user_id': user_id,
            'action': action,
            'details': details,
            'timestamp': time.time()
        }
        
        # Publish to user's followers
        for follower_id in self.get_followers(user_id):
            self.sse.publish(f'feed:{follower_id}', event_data, event='activity')
        
        # Publish to global feed
        self.sse.publish('feed:global', event_data, event='activity')
    
    def get_followers(self, user_id: str) -> list:
        # Fetch from database
        return db.get_followers(user_id)

# Usage
feed = ActivityFeedPublisher(redis_sse)

@app.route('/api/posts', methods=['POST'])
def create_post():
    post = create_post_in_db(request.json)
    
    # Publish activity
    feed.publish_activity(
        user_id=current_user.id,
        action='created_post',
        details={'post_id': post.id, 'title': post.title}
    )
    
    return jsonify(post.to_dict())
```

### Stock Price Ticker

```python
import asyncio
import random

class StockTickerPublisher:
    """Simulate stock price updates."""
    
    def __init__(self, sse_manager):
        self.sse = sse_manager
        self.stocks = {
            'AAPL': 150.00,
            'GOOGL': 2800.00,
            'MSFT': 300.00,
            'AMZN': 3400.00
        }
    
    async def start(self):
        while True:
            for symbol, price in self.stocks.items():
                # Simulate price change
                change = random.uniform(-0.5, 0.5)
                new_price = round(price + change, 2)
                self.stocks[symbol] = new_price
                
                self.sse.publish(
                    f'stocks:{symbol}',
                    {
                        'symbol': symbol,
                        'price': new_price,
                        'change': round(change, 2),
                        'change_percent': round(change / price * 100, 2)
                    },
                    event='price-update'
                )
            
            await asyncio.sleep(0.1)  # 10 updates/second
```

---

## Infrastructure Configuration

### Nginx

```nginx
location /events {
    proxy_pass http://backend;
    
    # SSE-specific settings
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    
    # Disable buffering
    proxy_buffering off;
    proxy_cache off;
    
    # Extended timeouts
    proxy_read_timeout 86400s;  # 24 hours
    proxy_send_timeout 86400s;
    
    # Chunked transfer encoding
    chunked_transfer_encoding on;
    
    # Disable nginx's event buffer
    proxy_set_header X-Accel-Buffering no;
}
```

### AWS ALB

```yaml
# CloudFormation
Resources:
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      # Extended idle timeout for SSE
      TargetGroupAttributes:
        - Key: deregistration_delay.timeout_seconds
          Value: '30'
        - Key: stickiness.enabled
          Value: 'true'
        - Key: stickiness.type
          Value: 'lb_cookie'
          
  Listener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref LoadBalancer
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup

  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      LoadBalancerAttributes:
        - Key: idle_timeout.timeout_seconds
          Value: '3600'  # 1 hour idle timeout
```

---

## Comparison with Alternatives

```
Feature              SSE          WebSocket     Long Polling
────────────────────────────────────────────────────────────
Direction            Server→Client  Bidirectional  Server→Client

Protocol             HTTP          WebSocket      HTTP

Reconnection         Automatic     Manual         Manual

Event Types          Built-in      Manual         Manual

Message ID/Replay    Built-in      Manual         Manual

Binary Data          No            Yes            Yes

Proxy/Firewall       Excellent     Good           Excellent

Complexity           Low           High           Medium

Browser Support      Good*         Excellent      Excellent

Max Connections      ~6/domain     Unlimited      ~6/domain

* IE not supported, use polyfill
```

---

## Key Takeaways

1. **Simple and standard**: SSE uses HTTP, works with existing infrastructure, and has built-in browser support

2. **Automatic reconnection**: EventSource API handles reconnection with Last-Event-ID for message recovery

3. **Event types**: Built-in support for named events enables routing different message types

4. **Unidirectional only**: Use WebSocket if you need bidirectional communication

5. **Disable buffering**: Configure nginx/proxies to disable response buffering for real-time delivery

6. **Scale with Redis**: Use pub/sub to broadcast events across multiple server instances

7. **Connection limits**: Browsers limit connections per domain (~6); consider HTTP/2 or connection pooling
