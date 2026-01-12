# Authentication Fundamentals

## TL;DR

Authentication verifies identity ("who are you?"). The challenge in distributed systems is doing this securely without sharing credentials everywhere, while handling session management at scale.

---

## The Problem Authentication Solves

In a monolithic application, authentication is simple:

```
1. User sends username + password
2. Server checks against database
3. Server creates session, stores in memory
4. Server returns session cookie
5. Subsequent requests include cookie
```

In distributed systems, this breaks down:

```
                    ┌─────────────┐
                    │   Load      │
     User ─────────►│  Balancer   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Server1 │    │ Server2 │    │ Server3 │
      │ Session │    │   ???   │    │   ???   │
      └─────────┘    └─────────┘    └─────────┘

Problem: Session created on Server1, but next
request goes to Server2 which has no session
```

---

## Session Management Strategies

### Strategy 1: Sticky Sessions

Load balancer routes all requests from same user to same server.

```
Implementation: Hash(user_id) → server

Pros:
- Simple implementation
- No shared state needed

Cons:
- Uneven load distribution
- Server failure loses all sessions
- Horizontal scaling is difficult
```

### Strategy 2: Centralized Session Store

All servers share a session store (Redis, Memcached).

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ session_id cookie
       ▼
┌─────────────┐
│   Server    │ ◄────┐
└──────┬──────┘      │
       │             │ Session data
       ▼             │
┌─────────────┐      │
│   Redis     │ ─────┘
│   Cluster   │
└─────────────┘
```

```python
# Session lookup on every request
def authenticate_request(request):
    session_id = request.cookies.get('session_id')
    if not session_id:
        return None
    
    # Hit Redis for every authenticated request
    session_data = redis.get(f"session:{session_id}")
    if not session_data:
        return None
    
    return json.loads(session_data)
```

**Trade-offs:**
- Pro: Any server can handle any request
- Con: Redis becomes single point of failure
- Con: Added latency for every request
- Con: Redis must scale with request rate, not user count

### Strategy 3: Stateless Tokens (JWT)

Encode session data in the token itself. Server validates without storage lookup.

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ Authorization: Bearer <JWT>
       ▼
┌─────────────┐
│   Server    │ Validates signature locally
└─────────────┘ No external lookup needed
```

**Trade-offs:**
- Pro: No session storage needed
- Pro: Scales infinitely
- Con: Cannot revoke tokens before expiry
- Con: Token size increases with claims

---

## Password Storage

### Never Store Plain Passwords

```python
# WRONG - attacker dumps database, gets all passwords
password_hash = hashlib.sha256(password).hexdigest()

# WRONG - rainbow table attack
password_hash = hashlib.sha256(password + "static_salt").hexdigest()

# CORRECT - unique salt per user, slow hash function
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
```

### Why Bcrypt/Argon2?

1. **Salted**: Each hash includes random salt
2. **Slow**: Configurable work factor
3. **CPU-intensive**: Resists GPU attacks (Argon2 is also memory-hard)

```python
# Verification
def verify_password(stored_hash, provided_password):
    return bcrypt.checkpw(
        provided_password.encode(),
        stored_hash.encode()
    )
```

### Work Factor Selection

| Work Factor | Time per Hash | Attempts/sec (attacker) |
|-------------|---------------|-------------------------|
| 10          | ~100ms        | 10                      |
| 12          | ~400ms        | 2.5                     |
| 14          | ~1.6s         | 0.6                     |

Choose factor that takes 250-500ms on your hardware.

---

## Multi-Factor Authentication (MFA)

### Something You Know + Something You Have

```
Factor 1: Password (knowledge)
Factor 2: One of:
  - TOTP code from authenticator app (possession)
  - SMS code (possession) - weaker, SIM swap attacks
  - Hardware key like YubiKey (possession)
  - Biometric (inherence)
```

### TOTP Implementation

```python
import pyotp
import time

# Setup: Generate secret, show QR code to user
secret = pyotp.random_base32()  # Store encrypted in DB
totp = pyotp.TOTP(secret)
provisioning_uri = totp.provisioning_uri(
    name="user@example.com",
    issuer_name="MyApp"
)
# Convert provisioning_uri to QR code for user to scan

# Verification
def verify_totp(user_secret, provided_code):
    totp = pyotp.TOTP(user_secret)
    # valid_window allows for clock drift
    return totp.verify(provided_code, valid_window=1)
```

### TOTP Internals

```
TOTP = HMAC-SHA1(secret, floor(time / 30))

Time:    1704067200  1704067230  1704067260
Code:    847293      159462      738291
         ◄─── 30s ──►◄─── 30s ──►
```

---

## Brute Force Protection

### Rate Limiting

```python
from redis import Redis
import time

def check_login_rate_limit(username, ip_address):
    redis = Redis()
    
    # Rate limit by username (prevents credential stuffing)
    user_key = f"login_attempts:user:{username}"
    user_attempts = redis.incr(user_key)
    redis.expire(user_key, 900)  # 15 minute window
    
    # Rate limit by IP (prevents distributed attacks)
    ip_key = f"login_attempts:ip:{ip_address}"
    ip_attempts = redis.incr(ip_key)
    redis.expire(ip_key, 3600)  # 1 hour window
    
    if user_attempts > 5:
        return False, "Too many attempts for this account"
    if ip_attempts > 20:
        return False, "Too many attempts from this IP"
    
    return True, None
```

### Progressive Delays

```python
def get_delay_after_failures(failure_count):
    """Exponential backoff with jitter"""
    if failure_count < 3:
        return 0
    
    base_delay = min(2 ** (failure_count - 2), 300)  # Max 5 minutes
    jitter = random.uniform(0, base_delay * 0.1)
    return base_delay + jitter
```

### Account Lockout

```
Attempt 1-3: Normal
Attempt 4-5: CAPTCHA required
Attempt 6-10: 15-minute soft lock
Attempt 11+: Account locked, email notification
```

---

## Credential Stuffing Defense

Attackers use breached password databases to try credentials on other sites.

### Detection Signals

```python
def calculate_risk_score(request, user):
    score = 0
    
    # New device
    if not is_known_device(user, request.device_fingerprint):
        score += 30
    
    # Unusual location
    if not is_usual_location(user, request.ip_address):
        score += 25
    
    # Unusual time
    if not is_usual_time(user, datetime.now()):
        score += 15
    
    # Failed attempts recently
    score += min(get_recent_failures(user) * 10, 30)
    
    return score

def handle_login(request, user, password_valid):
    risk_score = calculate_risk_score(request, user)
    
    if password_valid:
        if risk_score > 50:
            # Require step-up authentication
            return require_mfa(user)
        return success()
    else:
        if risk_score > 70:
            # Likely automated attack
            return temporary_block()
        return invalid_credentials()
```

### Have I Been Pwned Integration

```python
import hashlib
import requests

def is_password_breached(password):
    """Check against Have I Been Pwned API (k-anonymity)"""
    sha1 = hashlib.sha1(password.encode()).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]
    
    # Send only prefix to API
    response = requests.get(
        f"https://api.pwnedpasswords.com/range/{prefix}"
    )
    
    # Check if our suffix is in results
    for line in response.text.splitlines():
        hash_suffix, count = line.split(':')
        if hash_suffix == suffix:
            return True, int(count)
    
    return False, 0
```

---

## Session Security

### Secure Cookie Attributes

```python
response.set_cookie(
    'session_id',
    value=session_id,
    httponly=True,     # Not accessible via JavaScript
    secure=True,       # Only sent over HTTPS
    samesite='Lax',    # CSRF protection
    max_age=86400,     # 24 hours
    domain='.example.com',
    path='/'
)
```

### Session Fixation Prevention

```python
def login(user, password):
    if not verify_password(user, password):
        return error()
    
    # CRITICAL: Generate new session ID after authentication
    # Prevents attacker from setting session ID before login
    old_session_id = request.cookies.get('session_id')
    new_session_id = generate_secure_session_id()
    
    if old_session_id:
        redis.delete(f"session:{old_session_id}")
    
    redis.setex(
        f"session:{new_session_id}",
        86400,
        json.dumps({'user_id': user.id})
    )
    
    return response.set_cookie('session_id', new_session_id)
```

### Session Hijacking Prevention

```python
def validate_session(request):
    session = get_session(request)
    if not session:
        return None
    
    # Validate fingerprint hasn't changed
    current_fingerprint = generate_fingerprint(request)
    if session['fingerprint'] != current_fingerprint:
        # Possible session hijacking
        invalidate_session(session['id'])
        log_security_event('session_fingerprint_mismatch', session)
        return None
    
    return session

def generate_fingerprint(request):
    """Create fingerprint from stable request attributes"""
    components = [
        request.headers.get('User-Agent', ''),
        request.headers.get('Accept-Language', ''),
        # Don't include IP - changes with mobile/VPN
    ]
    return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]
```

---

## Authentication Flows

### Standard Login Flow

```
┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
│Client│      │Server│      │ Auth │      │  DB  │
└──┬───┘      └──┬───┘      │ Svc  │      └──┬───┘
   │             │          └──┬───┘         │
   │ POST /login │             │             │
   │ (user,pass) │             │             │
   │────────────►│             │             │
   │             │ Verify      │             │
   │             │────────────►│             │
   │             │             │ Get user    │
   │             │             │────────────►│
   │             │             │◄────────────│
   │             │             │ bcrypt      │
   │             │             │ verify      │
   │             │◄────────────│             │
   │             │ Create      │             │
   │             │ session     │             │
   │ Set-Cookie  │             │             │
   │◄────────────│             │             │
```

### Token Refresh Flow

```
Access Token:  Short-lived (15 min)
Refresh Token: Long-lived (7 days), stored securely

┌──────┐                      ┌──────┐
│Client│                      │Server│
└──┬───┘                      └──┬───┘
   │                             │
   │ Request + expired token     │
   │────────────────────────────►│
   │         401 Unauthorized    │
   │◄────────────────────────────│
   │                             │
   │ POST /refresh               │
   │ + refresh_token             │
   │────────────────────────────►│
   │                             │ Validate refresh token
   │                             │ Generate new access token
   │    New access token         │
   │◄────────────────────────────│
   │                             │
   │ Retry original request      │
   │────────────────────────────►│
```

---

## Single Sign-On (SSO) Overview

### Why SSO?

```
Without SSO:
User has credentials for: Email, CRM, HR System, Wiki, etc.
- Password fatigue → weak passwords
- Admin nightmare → provision/deprovision everywhere

With SSO:
User has one identity, accesses all systems
- One strong password + MFA
- Central access control
- Single audit log
```

### SSO Protocols

| Protocol | Use Case | Token Format |
|----------|----------|--------------|
| SAML 2.0 | Enterprise, legacy | XML |
| OAuth 2.0 | API authorization | JSON (JWT) |
| OpenID Connect | Modern authentication | JWT |
| LDAP/Kerberos | Internal/on-prem | Tickets |

---

## Trade-offs Summary

| Approach | Scalability | Revocation | Complexity |
|----------|-------------|------------|------------|
| Server Sessions | Low (sticky) | Instant | Low |
| Centralized Store | Medium | Instant | Medium |
| Stateless Tokens | High | Difficult | Medium |
| Hybrid (short JWT + refresh) | High | Near-instant | High |

---

## Security Checklist

```
□ Passwords hashed with bcrypt/Argon2 (cost factor ≥ 12)
□ HTTPS everywhere (HSTS enabled)
□ Secure cookie attributes (HttpOnly, Secure, SameSite)
□ Session regeneration on authentication state change
□ Rate limiting on authentication endpoints
□ Account lockout after failed attempts
□ MFA available (ideally required)
□ Breached password detection
□ Session timeout and absolute expiry
□ Audit logging of authentication events
```

---

## References

- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines (SP 800-63)](https://pages.nist.gov/800-63-3/)
- [Have I Been Pwned API](https://haveibeenpwned.com/API/v3)
- [RFC 6238: TOTP](https://datatracker.ietf.org/doc/html/rfc6238)
