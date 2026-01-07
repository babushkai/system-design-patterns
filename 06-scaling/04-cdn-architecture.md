# CDN Architecture

## TL;DR

A Content Delivery Network (CDN) distributes content across geographically dispersed edge servers, caching static and dynamic content close to users. This reduces latency, offloads origin servers, and improves availability. Modern CDNs also provide edge computing, security features, and real-time optimization.

---

## Why CDN?

Without CDN:

```
User in Tokyo                               Origin in Virginia
    │                                              │
    │◄────────── 200ms round trip ────────────────►│
    │                                              │
    │  Request: GET /image.png                     │
    │  ────────────────────────────────────────►   │
    │                                              │
    │  Response: 2MB image                         │
    │  ◄────────────────────────────────────────   │
    │                                              │
    │  Total: ~3 seconds for 2MB image             │
    │                                              │
```

With CDN:

```
User in Tokyo         Edge in Tokyo         Origin in Virginia
    │                      │                       │
    │◄──── 10ms ─────────►│                       │
    │                      │                       │
    │  Request             │                       │
    │  ──────────►         │                       │
    │                      │                       │
    │  Response (cached)   │                       │
    │  ◄──────────         │                       │
    │                      │                       │
    │  Total: ~100ms for 2MB image                 │
```

---

## CDN Architecture Overview

```
                              Internet Users
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
    ┌──────────┐              ┌──────────┐              ┌──────────┐
    │   PoP    │              │   PoP    │              │   PoP    │
    │  Tokyo   │              │  London  │              │  NYC     │
    │          │              │          │              │          │
    │ ┌──────┐ │              │ ┌──────┐ │              │ ┌──────┐ │
    │ │Edge  │ │              │ │Edge  │ │              │ │Edge  │ │
    │ │Server│ │              │ │Server│ │              │ │Server│ │
    │ └──────┘ │              │ └──────┘ │              │ └──────┘ │
    │ ┌──────┐ │              │ ┌──────┐ │              │ ┌──────┐ │
    │ │Cache │ │              │ │Cache │ │              │ │Cache │ │
    │ └──────┘ │              │ └──────┘ │              │ └──────┘ │
    └────┬─────┘              └────┬─────┘              └────┬─────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                          ┌────────┴────────┐
                          │  Origin Shield  │
                          │  (Mid-tier)     │
                          └────────┬────────┘
                                   │
                          ┌────────┴────────┐
                          │  Origin Server  │
                          │  (Your Server)  │
                          └─────────────────┘

PoP = Point of Presence
```

---

## CDN Request Flow

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time

class CacheStatus(Enum):
    HIT = "hit"           # Served from cache
    MISS = "miss"         # Not in cache, fetch from origin
    STALE = "stale"       # In cache but expired
    BYPASS = "bypass"     # Cache intentionally skipped

@dataclass
class CDNResponse:
    content: bytes
    status: CacheStatus
    age: int              # Seconds since cached
    ttl: int              # Remaining TTL
    edge_location: str

class EdgeServer:
    def __init__(self, location: str, origin_url: str):
        self.location = location
        self.origin = origin_url
        self.cache = {}  # url -> (content, cached_at, ttl)
    
    def handle_request(self, url: str, headers: dict) -> CDNResponse:
        # Check for cache bypass
        if headers.get('Cache-Control') == 'no-cache':
            return self._fetch_from_origin(url, CacheStatus.BYPASS)
        
        # Check local cache
        cached = self.cache.get(url)
        
        if cached:
            content, cached_at, ttl = cached
            age = int(time.time() - cached_at)
            
            if age < ttl:
                # Cache HIT
                return CDNResponse(
                    content=content,
                    status=CacheStatus.HIT,
                    age=age,
                    ttl=ttl - age,
                    edge_location=self.location
                )
            else:
                # Cache STALE - revalidate
                return self._revalidate(url, cached)
        
        # Cache MISS
        return self._fetch_from_origin(url, CacheStatus.MISS)
    
    def _fetch_from_origin(self, url: str, 
                           status: CacheStatus) -> CDNResponse:
        # Fetch from origin server
        response = self._origin_request(url)
        
        # Cache if cacheable
        if self._is_cacheable(response):
            ttl = self._get_ttl(response.headers)
            self.cache[url] = (response.content, time.time(), ttl)
        
        return CDNResponse(
            content=response.content,
            status=status,
            age=0,
            ttl=ttl,
            edge_location=self.location
        )
```

```
Request Flow Diagram:

┌─────────────────────────────────────────────────────────────┐
│                         USER REQUEST                         │
│                    GET /images/hero.jpg                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     DNS RESOLUTION                           │
│                                                              │
│   1. Query: cdn.example.com                                  │
│   2. GeoDNS returns nearest PoP IP                          │
│   3. Result: 203.0.113.45 (Tokyo PoP)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      EDGE SERVER                             │
│                                                              │
│   ┌───────────────────────────────────────────────────┐     │
│   │               CACHE LOOKUP                         │     │
│   │                    │                               │     │
│   │          ┌─────────┴─────────┐                    │     │
│   │          │                   │                    │     │
│   │        HIT               MISS/STALE               │     │
│   │          │                   │                    │     │
│   │          ▼                   ▼                    │     │
│   │    Return cached     Fetch from origin            │     │
│   │                      or shield                    │     │
│   └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Caching Strategies

### Cache-Control Headers

```python
from datetime import datetime, timedelta

class CacheControlBuilder:
    def __init__(self):
        self.directives = []
    
    def public(self):
        """Allow caching by CDN and browsers"""
        self.directives.append("public")
        return self
    
    def private(self):
        """Only allow browser caching, not CDN"""
        self.directives.append("private")
        return self
    
    def max_age(self, seconds: int):
        """Cache for this many seconds"""
        self.directives.append(f"max-age={seconds}")
        return self
    
    def s_maxage(self, seconds: int):
        """CDN-specific max age (overrides max-age for CDN)"""
        self.directives.append(f"s-maxage={seconds}")
        return self
    
    def stale_while_revalidate(self, seconds: int):
        """Serve stale while fetching fresh in background"""
        self.directives.append(f"stale-while-revalidate={seconds}")
        return self
    
    def stale_if_error(self, seconds: int):
        """Serve stale if origin returns error"""
        self.directives.append(f"stale-if-error={seconds}")
        return self
    
    def no_store(self):
        """Never cache"""
        self.directives.append("no-store")
        return self
    
    def build(self) -> str:
        return ", ".join(self.directives)

# Examples
static_assets = CacheControlBuilder()\
    .public()\
    .max_age(31536000)\  # 1 year
    .build()
# "public, max-age=31536000"

api_response = CacheControlBuilder()\
    .public()\
    .s_maxage(60)\
    .stale_while_revalidate(300)\
    .stale_if_error(86400)\
    .build()
# "public, s-maxage=60, stale-while-revalidate=300, stale-if-error=86400"

user_data = CacheControlBuilder()\
    .private()\
    .max_age(0)\
    .build()
# "private, max-age=0"
```

### Cache Key Design

```python
class CacheKeyBuilder:
    """
    Build cache keys that capture request variations
    """
    def __init__(self, base_url: str):
        self.base = base_url
        self.vary_headers = []
        self.query_params = []
        self.cookies = []
    
    def vary_on_header(self, header: str):
        """Include header value in cache key"""
        self.vary_headers.append(header)
        return self
    
    def vary_on_query(self, param: str):
        """Include query parameter in cache key"""
        self.query_params.append(param)
        return self
    
    def vary_on_cookie(self, cookie: str):
        """Include cookie in cache key"""
        self.cookies.append(cookie)
        return self
    
    def build_key(self, request) -> str:
        parts = [self.base]
        
        # Add selected headers
        for header in self.vary_headers:
            value = request.headers.get(header, '')
            parts.append(f"{header}={value}")
        
        # Add selected query params
        for param in self.query_params:
            value = request.args.get(param, '')
            parts.append(f"{param}={value}")
        
        # Add selected cookies
        for cookie in self.cookies:
            value = request.cookies.get(cookie, '')
            parts.append(f"{cookie}={value}")
        
        return "|".join(parts)

# Example: Different cache for mobile vs desktop
image_key = CacheKeyBuilder("/images/hero.jpg")\
    .vary_on_header("Accept")\        # WebP vs JPEG
    .vary_on_header("DPR")\           # Device pixel ratio
    .vary_on_query("width")\          # Responsive images
    .build_key(request)

# Cache keys:
# "/images/hero.jpg|Accept=image/webp|DPR=2|width=800"
# "/images/hero.jpg|Accept=image/jpeg|DPR=1|width=400"
```

---

## Origin Shield

```
Without Shield (Origin receives N requests per cache miss):

┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ Tokyo │ │London │ │ NYC   │ │Sydney │ │ Paris │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │         │
    │         │         │         │         │
    └─────────┴─────────┼─────────┴─────────┘
                        │
                        ▼
                  ┌──────────┐
                  │  Origin  │  ← 5 requests for same content!
                  └──────────┘


With Shield (Origin receives 1 request per cache miss):

┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ Tokyo │ │London │ │ NYC   │ │Sydney │ │ Paris │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │         │
    └─────────┴─────────┼─────────┴─────────┘
                        │
                        ▼
                  ┌──────────┐
                  │  Shield  │  ← Coalesces requests
                  │   PoP    │
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │  Origin  │  ← 1 request only
                  └──────────┘
```

```python
import asyncio
from collections import defaultdict
from typing import Awaitable

class OriginShield:
    def __init__(self):
        self.pending_requests = defaultdict(list)
        self.cache = {}
        self.locks = defaultdict(asyncio.Lock)
    
    async def get(self, url: str) -> bytes:
        # Check local cache first
        if url in self.cache:
            return self.cache[url]
        
        async with self.locks[url]:
            # Double-check after acquiring lock
            if url in self.cache:
                return self.cache[url]
            
            # Request coalescing: only one request to origin
            if not self.pending_requests[url]:
                # First request, fetch from origin
                future = asyncio.create_task(self._fetch_origin(url))
                self.pending_requests[url].append(future)
                
                try:
                    content = await future
                    self.cache[url] = content
                    return content
                finally:
                    del self.pending_requests[url]
            else:
                # Wait for existing request
                return await self.pending_requests[url][0]
    
    async def _fetch_origin(self, url: str) -> bytes:
        # Actual origin fetch
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.origin_url}{url}") as resp:
                return await resp.read()
```

---

## Cache Invalidation

### Purge by URL

```python
import aiohttp
from typing import List

class CDNPurger:
    def __init__(self, api_key: str, zone_id: str):
        self.api_key = api_key
        self.zone_id = zone_id
        self.api_url = "https://api.cloudflare.com/client/v4"
    
    async def purge_urls(self, urls: List[str]) -> dict:
        """Purge specific URLs from cache"""
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self.api_url}/zones/{self.zone_id}/purge_cache",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"files": urls}
            )
            return await resp.json()
    
    async def purge_by_prefix(self, prefix: str) -> dict:
        """Purge all URLs matching prefix"""
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self.api_url}/zones/{self.zone_id}/purge_cache",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"prefixes": [prefix]}
            )
            return await resp.json()
    
    async def purge_by_tag(self, tags: List[str]) -> dict:
        """Purge by cache tag (most efficient)"""
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self.api_url}/zones/{self.zone_id}/purge_cache",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"tags": tags}
            )
            return await resp.json()
    
    async def purge_all(self) -> dict:
        """Nuclear option - purge everything"""
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self.api_url}/zones/{self.zone_id}/purge_cache",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"purge_everything": True}
            )
            return await resp.json()

# Usage with cache tags
# Response headers: Cache-Tag: product-123, category-electronics
await purger.purge_by_tag(["product-123"])  # Invalidates all pages with this product
```

### Cache Versioning (URL-based invalidation)

```python
import hashlib
from flask import Flask, url_for

app = Flask(__name__)

class AssetVersioner:
    def __init__(self, static_folder: str):
        self.static_folder = static_folder
        self.version_cache = {}
    
    def get_versioned_url(self, filename: str) -> str:
        """Generate versioned URL based on file content"""
        if filename not in self.version_cache:
            filepath = f"{self.static_folder}/{filename}"
            with open(filepath, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()[:8]
            self.version_cache[filename] = content_hash
        
        version = self.version_cache[filename]
        return f"/static/{filename}?v={version}"
    
    def invalidate(self, filename: str):
        """Clear cached version (on file update)"""
        self.version_cache.pop(filename, None)

versioner = AssetVersioner("./static")

# In templates:
# <script src="{{ versioner.get_versioned_url('app.js') }}"></script>
# Outputs: <script src="/static/app.js?v=a1b2c3d4"></script>

# When file changes, hash changes = new URL = automatic cache bust
```

---

## Edge Computing

```python
# Cloudflare Workers example
"""
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // A/B testing at the edge
  const variant = request.headers.get('cookie')?.includes('variant=b') 
    ? 'b' 
    : 'a'
  
  // Modify request before sending to origin
  const modifiedUrl = new URL(request.url)
  modifiedUrl.pathname = `/${variant}${url.pathname}`
  
  const response = await fetch(modifiedUrl, request)
  
  // Modify response
  const newResponse = new Response(response.body, response)
  newResponse.headers.set('X-Variant', variant)
  
  // Set variant cookie for consistency
  if (!request.headers.get('cookie')?.includes('variant=')) {
    newResponse.headers.set('Set-Cookie', `variant=${variant}; max-age=86400`)
  }
  
  return newResponse
}
"""

# Python equivalent for edge logic
class EdgeHandler:
    async def handle_request(self, request: dict) -> dict:
        url = request['url']
        headers = request['headers']
        
        # Geolocation-based routing
        country = headers.get('CF-IPCountry', 'US')
        
        if country in ['CN', 'HK', 'TW']:
            # Route to APAC origin
            origin = "https://apac-origin.example.com"
        elif country in ['DE', 'FR', 'GB']:
            # Route to EU origin
            origin = "https://eu-origin.example.com"
        else:
            # Default to US origin
            origin = "https://us-origin.example.com"
        
        # Bot detection
        user_agent = headers.get('User-Agent', '')
        if self._is_bot(user_agent):
            return self._serve_bot_response(request)
        
        # Image optimization
        if url.endswith(('.jpg', '.png', '.webp')):
            return await self._optimize_image(request, origin)
        
        # Forward to origin
        return await self._forward_to_origin(request, origin)
    
    def _is_bot(self, user_agent: str) -> bool:
        bot_patterns = ['Googlebot', 'Bingbot', 'curl', 'wget']
        return any(bot in user_agent for bot in bot_patterns)
    
    async def _optimize_image(self, request: dict, origin: str) -> dict:
        # Check for WebP support
        accept = request['headers'].get('Accept', '')
        supports_webp = 'image/webp' in accept
        
        # Get device pixel ratio
        dpr = float(request['headers'].get('DPR', '1'))
        
        # Build optimized image URL
        width = request.get('width', 800)
        format_ext = 'webp' if supports_webp else 'jpg'
        
        optimized_url = f"{origin}/cdn-cgi/image/width={int(width*dpr)},format={format_ext}/{request['url']}"
        
        return await self._fetch(optimized_url)
```

---

## Multi-CDN Architecture

```
                            ┌─────────────────┐
                            │  DNS/Traffic    │
                            │   Manager       │
                            │  (Route53/NS1)  │
                            └────────┬────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  CloudFlare  │          │   Fastly     │          │  Akamai      │
    │    (60%)     │          │    (25%)     │          │   (15%)      │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                              ┌──────┴───────┐
                              │    Origin    │
                              └──────────────┘
```

```python
import random
from dataclasses import dataclass
from typing import List, Dict
import asyncio

@dataclass
class CDNProvider:
    name: str
    weight: int  # Traffic percentage
    health_url: str
    base_url: str
    is_healthy: bool = True

class MultiCDNRouter:
    def __init__(self, providers: List[CDNProvider]):
        self.providers = providers
        self._build_weighted_list()
    
    def _build_weighted_list(self):
        """Build weighted selection list"""
        self.weighted_providers = []
        for provider in self.providers:
            if provider.is_healthy:
                self.weighted_providers.extend(
                    [provider] * provider.weight
                )
    
    def select_cdn(self) -> CDNProvider:
        """Select CDN based on weights"""
        if not self.weighted_providers:
            raise Exception("No healthy CDN providers")
        return random.choice(self.weighted_providers)
    
    async def health_check_loop(self):
        """Continuously monitor CDN health"""
        while True:
            for provider in self.providers:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            provider.health_url, 
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            was_healthy = provider.is_healthy
                            provider.is_healthy = resp.status == 200
                            
                            if was_healthy != provider.is_healthy:
                                self._build_weighted_list()
                                print(f"CDN {provider.name} health: {provider.is_healthy}")
                except Exception:
                    if provider.is_healthy:
                        provider.is_healthy = False
                        self._build_weighted_list()
            
            await asyncio.sleep(10)
    
    def get_failover_provider(self, 
                               failed_provider: CDNProvider) -> CDNProvider:
        """Get alternative provider on failure"""
        for provider in self.providers:
            if provider != failed_provider and provider.is_healthy:
                return provider
        raise Exception("No failover CDN available")

# Usage
router = MultiCDNRouter([
    CDNProvider("cloudflare", 60, "https://cf.example.com/health", 
                "https://cf.example.com"),
    CDNProvider("fastly", 25, "https://fastly.example.com/health",
                "https://fastly.example.com"),
    CDNProvider("akamai", 15, "https://akamai.example.com/health",
                "https://akamai.example.com"),
])
```

---

## Performance Metrics

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class CDNMetrics:
    # Cache performance
    cache_hit_ratio: float    # Target: > 90%
    cache_miss_ratio: float
    cache_bypass_ratio: float
    
    # Latency
    edge_latency_p50: float   # Target: < 50ms
    edge_latency_p99: float   # Target: < 200ms
    origin_latency_p50: float
    
    # Bandwidth
    bandwidth_saved_gb: float
    origin_bandwidth_gb: float
    
    # Errors
    error_rate_4xx: float     # Target: < 1%
    error_rate_5xx: float     # Target: < 0.1%

class CDNMonitor:
    def __init__(self, cdn_api):
        self.api = cdn_api
    
    async def collect_metrics(self, time_range: str = "1h") -> CDNMetrics:
        data = await self.api.get_analytics(time_range)
        
        total_requests = data['requests']['total']
        cache_hits = data['requests']['cached']
        
        return CDNMetrics(
            cache_hit_ratio=cache_hits / total_requests,
            cache_miss_ratio=(total_requests - cache_hits) / total_requests,
            cache_bypass_ratio=data['requests']['bypass'] / total_requests,
            edge_latency_p50=data['latency']['edge']['p50'],
            edge_latency_p99=data['latency']['edge']['p99'],
            origin_latency_p50=data['latency']['origin']['p50'],
            bandwidth_saved_gb=data['bandwidth']['cached'] / 1e9,
            origin_bandwidth_gb=data['bandwidth']['uncached'] / 1e9,
            error_rate_4xx=data['errors']['4xx'] / total_requests,
            error_rate_5xx=data['errors']['5xx'] / total_requests,
        )
    
    def calculate_cost_savings(self, metrics: CDNMetrics, 
                                origin_cost_per_gb: float = 0.09) -> dict:
        """Calculate cost savings from CDN caching"""
        bandwidth_cost_saved = metrics.bandwidth_saved_gb * origin_cost_per_gb
        origin_requests_avoided = metrics.cache_hit_ratio
        
        return {
            "bandwidth_cost_saved_usd": bandwidth_cost_saved,
            "origin_load_reduction_pct": metrics.cache_hit_ratio * 100,
            "effective_origin_capacity_multiplier": 1 / (1 - metrics.cache_hit_ratio)
        }
```

---

## CDN Provider Comparison

| Feature | CloudFlare | Fastly | Akamai | AWS CloudFront |
|---------|------------|--------|--------|----------------|
| Global PoPs | 285+ | 80+ | 4000+ | 450+ |
| Edge Compute | Workers | Compute@Edge | EdgeWorkers | Lambda@Edge |
| Instant Purge | Yes | Yes (<150ms) | No (~5s) | Yes (~1min) |
| Free Tier | Generous | Limited | No | Limited |
| WebSocket | Yes | Yes | Yes | Yes |
| Real-time logs | Yes | Yes | Yes | Yes |

---

## Key Takeaways

1. **Cache everything possible**: Static assets, API responses, HTML pages—the more you cache, the better performance and lower costs

2. **Use appropriate TTLs**: Long TTLs (1 year) for versioned static assets; short TTLs with stale-while-revalidate for dynamic content

3. **Implement cache tags**: Enable surgical purging without purging unrelated content

4. **Deploy origin shield**: Reduces origin load dramatically, especially for cache misses across multiple PoPs

5. **Consider multi-CDN**: Critical for high-availability; use active-active or failover configurations

6. **Leverage edge computing**: Move logic closer to users for authentication, A/B testing, personalization

7. **Monitor cache hit ratio**: Aim for >90%; investigate patterns causing cache misses
