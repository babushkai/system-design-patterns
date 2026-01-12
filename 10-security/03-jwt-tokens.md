# JSON Web Tokens (JWT)

## TL;DR

JWTs are self-contained tokens that encode claims as JSON, signed to ensure integrity. They enable stateless authentication but come with trade-offs around revocation and size. Most JWT security issues stem from implementation errors, not protocol flaws.

---

## JWT Structure

A JWT consists of three base64url-encoded parts separated by dots:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Header.Payload.Signature
```

### Header

```json
{
    "alg": "HS256",    // Signing algorithm
    "typ": "JWT"       // Token type
}
```

Common algorithms:
- **HS256**: HMAC with SHA-256 (symmetric)
- **RS256**: RSA signature with SHA-256 (asymmetric)
- **ES256**: ECDSA with SHA-256 (asymmetric)

### Payload (Claims)

```json
{
    "iss": "https://auth.example.com",  // Issuer
    "sub": "user_12345",                // Subject (user ID)
    "aud": "my_api",                    // Audience
    "exp": 1704067200,                  // Expiration (Unix timestamp)
    "iat": 1704063600,                  // Issued at
    "nbf": 1704063600,                  // Not valid before
    "jti": "unique-token-id",           // JWT ID (for revocation)
    
    // Custom claims
    "role": "admin",
    "permissions": ["read", "write"]
}
```

### Signature

```
HMACSHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    secret
)
```

---

## Symmetric vs. Asymmetric Signing

### Symmetric (HS256)

Same secret for signing and verification.

```
┌─────────────────┐         ┌─────────────────┐
│  Auth Server    │         │  API Server     │
│  (signs JWT)    │         │  (verifies JWT) │
│                 │         │                 │
│  secret: xyz    │         │  secret: xyz    │
└─────────────────┘         └─────────────────┘

Problem: Every service that verifies needs the secret
         If any service is compromised, attacker can forge tokens
```

### Asymmetric (RS256/ES256)

Private key signs, public key verifies.

```
┌─────────────────┐         ┌─────────────────┐
│  Auth Server    │         │  API Server     │
│  (signs JWT)    │         │  (verifies JWT) │
│                 │         │                 │
│  PRIVATE key    │         │  PUBLIC key     │
│  (kept secret)  │         │  (shareable)    │
└─────────────────┘         └─────────────────┘

Advantage: 
- Only auth server can create tokens
- Compromised API server can't forge tokens
- Public keys can be published via JWKS
```

### When to Use Which

| Scenario | Recommendation |
|----------|----------------|
| Single monolithic app | HS256 (simpler) |
| Microservices | RS256/ES256 |
| Third-party integration | RS256/ES256 |
| High-security environments | ES256 (smaller, faster) |

---

## Creating JWTs

### Python Example (PyJWT)

```python
import jwt
import datetime

# Symmetric (HS256)
def create_jwt_symmetric(user_id, secret):
    payload = {
        'sub': user_id,
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        'iss': 'my-auth-server',
        'aud': 'my-api'
    }
    return jwt.encode(payload, secret, algorithm='HS256')

# Asymmetric (RS256)
def create_jwt_asymmetric(user_id, private_key):
    payload = {
        'sub': user_id,
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        'iss': 'my-auth-server',
        'aud': 'my-api'
    }
    return jwt.encode(payload, private_key, algorithm='RS256')
```

### Node.js Example (jsonwebtoken)

```javascript
const jwt = require('jsonwebtoken');

// Create token
const token = jwt.sign(
    { 
        sub: 'user_123',
        role: 'admin'
    },
    process.env.JWT_SECRET,
    {
        algorithm: 'HS256',
        expiresIn: '1h',
        issuer: 'my-auth-server',
        audience: 'my-api'
    }
);
```

---

## Validating JWTs

### Validation Checklist

```python
def validate_jwt(token, public_key_or_secret):
    try:
        # 1. Decode and verify signature
        decoded = jwt.decode(
            token,
            public_key_or_secret,
            algorithms=['RS256'],  # Explicitly specify allowed algorithms!
            audience='my-api',
            issuer='my-auth-server',
            options={
                'require': ['exp', 'iat', 'sub']  # Required claims
            }
        )
        
        # 2. Additional business logic validation
        if decoded.get('role') not in ['admin', 'user']:
            raise ValueError("Invalid role")
        
        return decoded
        
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired")
    except jwt.InvalidAudienceError:
        raise AuthError("Invalid audience")
    except jwt.InvalidIssuerError:
        raise AuthError("Invalid issuer")
    except jwt.InvalidSignatureError:
        raise AuthError("Invalid signature")
```

### Critical: Always Specify Algorithm

```python
# VULNERABLE - attacker can use 'none' algorithm
decoded = jwt.decode(token, secret)

# VULNERABLE - attacker can switch RS256 to HS256
decoded = jwt.decode(token, public_key, algorithms=['RS256', 'HS256'])

# SECURE - explicitly allow only expected algorithm
decoded = jwt.decode(token, public_key, algorithms=['RS256'])
```

---

## JWT Security Vulnerabilities

### 1. Algorithm Confusion Attack

```
Attack scenario:
1. Server expects RS256 (asymmetric)
2. Attacker takes PUBLIC key (which is public)
3. Attacker creates token with alg=HS256
4. Attacker signs with public key as HMAC secret
5. Server (misconfigured) verifies HS256 using public key as secret
6. Signature matches! Attacker forges tokens.

Prevention:
- NEVER accept algorithm from token header
- Always specify expected algorithm in verification
```

### 2. None Algorithm Attack

```
Attack scenario:
1. Attacker sets header: {"alg": "none"}
2. Attacker removes signature
3. Poorly configured library accepts unsigned token

Prevention:
- Explicitly specify algorithms=['RS256'] in decode
- Never include 'none' in allowed algorithms
```

### 3. Weak Secrets

```python
# BAD - easily brute-forced
secret = "secret"
secret = "password123"

# GOOD - cryptographically random
import secrets
secret = secrets.token_hex(32)  # 256 bits
```

**Brute Force Reality:**

```
Secret length  | Time to crack (modern GPU)
---------------|---------------------------
8 chars        | Seconds to minutes
16 chars       | Days to weeks
32 chars       | Computationally infeasible
```

### 4. Token Stored in Vulnerable Location

```javascript
// BAD - XSS can steal token
localStorage.setItem('token', jwt);

// BAD - Same issue
sessionStorage.setItem('token', jwt);

// BETTER - Not accessible via JavaScript
// Set via HttpOnly cookie from server

// BEST - Keep in memory, use refresh token rotation
let accessToken = null; // In-memory only
```

### 5. No Expiration or Too Long

```python
# BAD - No expiration
payload = {'sub': 'user123'}

# BAD - 30-day access token
payload = {'sub': 'user123', 'exp': now + timedelta(days=30)}

# GOOD - Short-lived access token
payload = {'sub': 'user123', 'exp': now + timedelta(minutes=15)}
```

---

## Token Revocation Strategies

JWTs are stateless - by design, you can't revoke them. Here are workarounds:

### Strategy 1: Short Expiration + Refresh Tokens

```
Access Token:  15 minutes
Refresh Token: 7 days (stored in DB, revocable)

Flow:
1. User logs out
2. Delete refresh token from DB
3. Access token still valid for up to 15 min (acceptable)
4. After 15 min, refresh fails, user must re-login
```

### Strategy 2: Token Blacklist

```python
# Redis-based blacklist
def revoke_token(jti):
    # Store with TTL matching token expiration
    token_exp = get_token_expiration(jti)
    ttl = token_exp - time.time()
    redis.setex(f"blacklist:{jti}", int(ttl), "revoked")

def is_token_revoked(jti):
    return redis.exists(f"blacklist:{jti}")

def validate_token(token):
    decoded = jwt.decode(token, ...)
    if is_token_revoked(decoded['jti']):
        raise AuthError("Token has been revoked")
    return decoded
```

**Trade-off:** Adds database lookup to every request, partially negating stateless benefit.

### Strategy 3: Token Versioning

```python
# Store token version per user in DB/cache
# When user logs out or changes password, increment version

def create_token(user):
    return jwt.encode({
        'sub': user.id,
        'token_version': user.token_version,  # Include current version
        'exp': ...
    }, secret)

def validate_token(token):
    decoded = jwt.decode(token, ...)
    user = get_user(decoded['sub'])
    
    # Check if token version matches
    if decoded['token_version'] != user.token_version:
        raise AuthError("Token has been invalidated")
    
    return decoded
```

### Strategy 4: Hybrid Approach

```
Short-lived JWT (15 min) for most requests
  ↓ Expired?
Refresh with refresh token (checked against DB)
  ↓ Valid?
Issue new access token
  ↓ Invalid?
Force re-authentication

Critical actions (password change, payment):
  - Always verify against DB regardless of JWT validity
```

---

## JWT Size Considerations

JWTs can get large, impacting performance.

### Size Breakdown

```
Typical JWT:
  Header:    ~36 bytes (base64)
  Payload:   ~200-500 bytes (base64)
  Signature: ~86 bytes (RS256) or ~43 bytes (HS256)
  Total:     ~300-700 bytes

Problematic JWT (too many claims):
  Payload with roles, permissions, user data: 2-4 KB
```

### Size Impact

```
Every HTTP request includes JWT in header:
  Authorization: Bearer <token>

If token is 2KB and user makes 100 requests:
  200KB of bandwidth just for tokens

Mobile/slow networks: Significant latency impact
```

### Size Reduction Strategies

```python
# BAD - Embedding all user data
payload = {
    'sub': 'user123',
    'name': 'John Doe',
    'email': 'john@example.com',
    'address': {...},
    'permissions': ['read:users', 'write:users', ...],  # 50 permissions
    'roles': ['admin', 'manager', ...],
}

# GOOD - Minimal claims, fetch details when needed
payload = {
    'sub': 'user123',
    'role': 'admin',  # Single role, not list
    'exp': ...
}
# Fetch full permissions from cache/DB when needed
```

---

## Access Token vs. ID Token

### Access Token

- **Purpose:** Authorize access to resources
- **Audience:** Resource server (API)
- **Contents:** Permissions, scopes
- **Validation:** Resource server validates
- **Opacity:** Can be opaque (not JWT) or JWT

### ID Token (OpenID Connect)

- **Purpose:** Authenticate user identity
- **Audience:** Client application
- **Contents:** User identity claims
- **Validation:** Client validates
- **Format:** Always JWT

```python
# Access token - for API calls
headers = {'Authorization': f'Bearer {access_token}'}
response = requests.get('https://api.example.com/data', headers=headers)

# ID token - for getting user info in client
id_token_claims = jwt.decode(id_token, ...)
user_email = id_token_claims['email']
```

**Important:** Never send ID token to resource servers. It's not for authorization.

---

## Implementation Patterns

### Middleware Pattern

```python
# Flask example
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'Missing token'}), 401
        
        try:
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=['RS256'],
                audience='my-api'
            )
            request.user = decoded
        except jwt.InvalidTokenError as e:
            return jsonify({'error': str(e)}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/protected')
@require_auth
def protected():
    return jsonify({'user': request.user['sub']})
```

### Scope-Based Authorization

```python
def require_scope(required_scope):
    def decorator(f):
        @wraps(f)
        @require_auth  # First authenticate
        def decorated(*args, **kwargs):
            user_scopes = request.user.get('scope', '').split()
            
            if required_scope not in user_scopes:
                return jsonify({'error': 'Insufficient scope'}), 403
            
            return f(*args, **kwargs)
        return decorated
    return decorator

@app.route('/admin')
@require_scope('admin:read')
def admin_endpoint():
    return jsonify({'admin': True})
```

---

## Testing JWTs

### Generating Test Tokens

```python
import jwt
from datetime import datetime, timedelta

def create_test_token(claims_override=None):
    claims = {
        'sub': 'test_user',
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=1),
        'iss': 'test-issuer',
        'aud': 'test-audience'
    }
    claims.update(claims_override or {})
    return jwt.encode(claims, 'test-secret', algorithm='HS256')

# Test cases
def test_expired_token():
    token = create_test_token({
        'exp': datetime.utcnow() - timedelta(hours=1)
    })
    response = client.get('/protected', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 401

def test_wrong_audience():
    token = create_test_token({'aud': 'wrong-audience'})
    response = client.get('/protected', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 401
```

### JWT Debugging

```bash
# Decode JWT without verification (for debugging only!)
echo "eyJhbGciOiJIUzI1NiIs..." | cut -d. -f2 | base64 -d | jq

# Or use jwt.io (NEVER paste production tokens!)
```

---

## Best Practices Summary

```
Token Creation:
□ Use RS256/ES256 for distributed systems
□ Include standard claims (iss, sub, aud, exp, iat)
□ Keep payload minimal
□ Use cryptographically strong secrets (≥256 bits)
□ Short expiration (15 min for access tokens)

Token Validation:
□ Always specify allowed algorithms explicitly
□ Validate all standard claims (iss, aud, exp)
□ Use constant-time comparison for signatures
□ Handle validation errors gracefully

Storage:
□ Never store in localStorage/sessionStorage
□ Use HttpOnly cookies or in-memory storage
□ Implement secure refresh token rotation

Revocation:
□ Implement refresh token rotation
□ Consider token blacklist for critical apps
□ Increment token version on security events
```

---

## References

- [RFC 7519: JSON Web Token](https://datatracker.ietf.org/doc/html/rfc7519)
- [RFC 7515: JSON Web Signature](https://datatracker.ietf.org/doc/html/rfc7515)
- [RFC 7518: JSON Web Algorithms](https://datatracker.ietf.org/doc/html/rfc7518)
- [JWT Best Practices (Auth0)](https://auth0.com/blog/jwt-security-best-practices/)
- [Critical vulnerabilities in JSON Web Token libraries](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/)
