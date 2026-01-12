# OAuth 2.0 and OpenID Connect

## TL;DR

OAuth 2.0 is for **authorization** (granting access to resources). OpenID Connect (OIDC) adds **authentication** (verifying identity) on top of OAuth 2.0. Together they enable secure, delegated access without sharing passwords.

---

## The Problem OAuth Solves

Before OAuth, granting a third-party app access to your data required sharing your password:

```
Scenario: Photo printing service needs access to your Google Photos

Old way (anti-pattern):
1. Give photo service your Google password
2. Photo service logs into Google as you
3. Photo service has full access to everything

Problems:
- Photo service can read your email, delete files
- You can't revoke access without changing password
- Password stored in multiple places
```

OAuth solution: Delegate limited, revocable access without sharing credentials.

---

## OAuth 2.0 Roles

```
┌─────────────────┐
│ Resource Owner  │  You (the human user)
└────────┬────────┘
         │ Grants permission
         ▼
┌─────────────────┐         ┌─────────────────┐
│     Client      │────────►│    Resource     │
│ (Photo Print)   │ Access  │     Server      │
└─────────────────┘ Token   │ (Google Photos) │
         ▲                  └─────────────────┘
         │ Issues token
┌────────┴────────┐
│ Authorization   │
│     Server      │
│ (Google OAuth)  │
└─────────────────┘
```

| Role | Description | Example |
|------|-------------|---------|
| Resource Owner | Entity that can grant access | User |
| Client | App requesting access | Photo printing app |
| Resource Server | API hosting protected resources | Google Photos API |
| Authorization Server | Issues tokens after authentication | Google OAuth |

---

## OAuth 2.0 Grant Types

### Authorization Code Grant (Recommended for Web Apps)

Most secure for server-side applications with confidential clients.

```
┌──────┐          ┌──────┐          ┌────────┐          ┌────────┐
│ User │          │Client│          │  Auth  │          │Resource│
│      │          │ App  │          │ Server │          │ Server │
└──┬───┘          └──┬───┘          └───┬────┘          └───┬────┘
   │                 │                  │                   │
   │ Click "Login    │                  │                   │
   │ with Google"    │                  │                   │
   │────────────────►│                  │                   │
   │                 │                  │                   │
   │    Redirect to authorization endpoint                  │
   │◄────────────────┤                  │                   │
   │                 │                  │                   │
   │ Navigate to auth server            │                   │
   │───────────────────────────────────►│                   │
   │                 │                  │                   │
   │         Login + Consent Screen     │                   │
   │◄──────────────────────────────────►│                   │
   │                 │                  │                   │
   │ Redirect with authorization code   │                   │
   │◄───────────────────────────────────┤                   │
   │                 │                  │                   │
   │ Navigate back   │                  │                   │
   │────────────────►│                  │                   │
   │                 │                  │                   │
   │                 │ Exchange code    │                   │
   │                 │ + client_secret  │                   │
   │                 │ for tokens       │                   │
   │                 │─────────────────►│                   │
   │                 │                  │                   │
   │                 │ Access Token +   │                   │
   │                 │ Refresh Token    │                   │
   │                 │◄─────────────────┤                   │
   │                 │                  │                   │
   │                 │           Access API with token      │
   │                 │─────────────────────────────────────►│
   │                 │                  │                   │
   │                 │              Protected Resource      │
   │                 │◄─────────────────────────────────────│
```

**Step 1: Authorization Request**

```
GET https://auth.example.com/authorize?
    response_type=code
    &client_id=photo_print_app
    &redirect_uri=https://photoprint.com/callback
    &scope=photos.read
    &state=xyz123  # CSRF protection
```

**Step 2: User Grants Permission**

```
Authorization Server shows:
┌─────────────────────────────────────┐
│  Photo Print App wants to:          │
│                                     │
│  ☑ View your photos                 │
│                                     │
│  [Allow]  [Deny]                    │
└─────────────────────────────────────┘
```

**Step 3: Redirect with Code**

```
HTTP 302 Found
Location: https://photoprint.com/callback?
    code=SplxlOBeZQQYbYS6WxSbIA
    &state=xyz123
```

**Step 4: Exchange Code for Token (Server-to-Server)**

```
POST https://auth.example.com/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=SplxlOBeZQQYbYS6WxSbIA
&redirect_uri=https://photoprint.com/callback
&client_id=photo_print_app
&client_secret=secret123
```

**Response:**

```json
{
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "tGzv3JOkF0XG5Qx2TlKWIA",
    "scope": "photos.read"
}
```

### Authorization Code with PKCE (Public Clients)

For mobile/SPA apps that can't keep a client_secret.

```
PKCE = Proof Key for Code Exchange

Before authorization:
1. Generate random code_verifier (43-128 chars)
2. Create code_challenge = SHA256(code_verifier)

Authorization request includes:
  &code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
  &code_challenge_method=S256

Token exchange includes:
  &code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk

Server verifies: SHA256(code_verifier) == code_challenge
```

**Why PKCE Matters:**

```
Without PKCE (vulnerable):
┌──────────┐                    ┌──────────┐
│ Attacker │                    │   App    │
└────┬─────┘                    └────┬─────┘
     │ Intercept redirect           │
     │ (malware, URL scheme)        │
     │◄────────────────────────────►│
     │                              │
     │ Exchange intercepted code    │
     │ for token                    │
     ▼                              │
 ATTACKER HAS TOKEN                 │

With PKCE (secure):
     │ Intercept redirect           │
     │◄────────────────────────────►│
     │                              │
     │ Exchange code BUT            │
     │ don't know code_verifier     │
     │                              │
     ▼                              │
 EXCHANGE FAILS                     │
```

### Client Credentials Grant (Machine-to-Machine)

No user involved. Service authenticates directly.

```
POST https://auth.example.com/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=backend_service
&client_secret=secret123
&scope=internal.read
```

**Use Case:** Microservice A calls Microservice B.

### Refresh Token Grant

Exchange refresh token for new access token without user interaction.

```
POST https://auth.example.com/token

grant_type=refresh_token
&refresh_token=tGzv3JOkF0XG5Qx2TlKWIA
&client_id=photo_print_app
&client_secret=secret123
```

---

## OpenID Connect (OIDC)

OIDC adds an identity layer on top of OAuth 2.0.

### OAuth 2.0 vs. OpenID Connect

```
OAuth 2.0:
  Q: "Can this app access my photos?"
  A: "Yes, here's an access token"
  
OpenID Connect:
  Q: "Who is this user?"
  A: "Here's an ID token with user info"
```

### The ID Token

A JWT containing claims about the authenticated user.

```json
{
    "iss": "https://auth.example.com",
    "sub": "user_12345",
    "aud": "photo_print_app",
    "exp": 1704067200,
    "iat": 1704063600,
    "nonce": "abc123",
    "name": "John Doe",
    "email": "john@example.com",
    "email_verified": true,
    "picture": "https://example.com/john.jpg"
}
```

**Key Claims:**

| Claim | Description |
|-------|-------------|
| iss | Issuer (who issued the token) |
| sub | Subject (unique user identifier) |
| aud | Audience (intended recipient) |
| exp | Expiration time |
| iat | Issued at time |
| nonce | Replay attack prevention |

### OIDC Flow

```
Authorization Request:
GET https://auth.example.com/authorize?
    response_type=code
    &client_id=photo_print_app
    &redirect_uri=https://photoprint.com/callback
    &scope=openid profile email  ← OIDC scopes
    &state=xyz123
    &nonce=abc456  ← For ID token validation
```

**Standard OIDC Scopes:**

| Scope | Claims Returned |
|-------|-----------------|
| openid | sub (required for OIDC) |
| profile | name, family_name, picture, etc. |
| email | email, email_verified |
| address | address |
| phone | phone_number, phone_number_verified |

### UserInfo Endpoint

Get additional user info with access token:

```
GET https://auth.example.com/userinfo
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...

Response:
{
    "sub": "user_12345",
    "name": "John Doe",
    "email": "john@example.com",
    "picture": "https://example.com/john.jpg"
}
```

---

## Token Validation

### Access Token Validation

**Option 1: Introspection (Opaque Tokens)**

```
POST https://auth.example.com/introspect
Authorization: Basic base64(client_id:client_secret)

token=eyJhbGciOiJSUzI1NiIs...
```

```json
{
    "active": true,
    "client_id": "photo_print_app",
    "username": "john",
    "scope": "photos.read",
    "exp": 1704067200
}
```

**Option 2: Local Validation (JWTs)**

```python
import jwt
from jwt import PyJWKClient

def validate_access_token(token):
    # Fetch public keys from JWKS endpoint
    jwks_client = PyJWKClient("https://auth.example.com/.well-known/jwks.json")
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    
    # Verify signature and claims
    decoded = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience="my_api",
        issuer="https://auth.example.com"
    )
    
    return decoded
```

### ID Token Validation Checklist

```python
def validate_id_token(id_token, expected_nonce):
    decoded = jwt.decode(id_token, options={"verify_signature": True})
    
    # 1. Verify issuer matches your auth server
    assert decoded['iss'] == "https://auth.example.com"
    
    # 2. Verify audience is your client_id
    assert decoded['aud'] == "photo_print_app"
    
    # 3. Verify token is not expired
    assert decoded['exp'] > time.time()
    
    # 4. Verify nonce matches (prevents replay)
    assert decoded['nonce'] == expected_nonce
    
    # 5. Verify signature (done by jwt.decode with verify_signature=True)
    
    return decoded
```

---

## Token Storage Best Practices

### Web Applications

```
Access Token:
  - Store in memory (JavaScript variable)
  - NOT in localStorage (XSS vulnerable)
  - NOT in sessionStorage (XSS vulnerable)
  
Refresh Token:
  - Store in HttpOnly, Secure cookie
  - Or use Backend-for-Frontend (BFF) pattern
```

### Backend-for-Frontend (BFF) Pattern

```
┌──────────────────────────────────────────────────────┐
│                     Browser                          │
│  ┌─────────────────────────────────────────────┐    │
│  │              SPA (React/Vue)                │    │
│  │  - Only has session cookie                  │    │
│  │  - No tokens in JavaScript                  │    │
│  └─────────────────────────────────────────────┘    │
└────────────────────────┬─────────────────────────────┘
                         │ Session cookie
                         ▼
              ┌─────────────────────┐
              │   BFF (Backend)     │
              │  - Holds tokens     │
              │  - Manages refresh  │
              └──────────┬──────────┘
                         │ Access Token
                         ▼
              ┌─────────────────────┐
              │   Resource Server   │
              └─────────────────────┘
```

### Mobile Applications

```
Refresh Token:
  - iOS: Keychain (kSecClassGenericPassword)
  - Android: EncryptedSharedPreferences or Keystore
  
Access Token:
  - Memory only, re-fetch on app restart
```

---

## Common Security Pitfalls

### 1. Missing State Parameter (CSRF)

```python
# BAD: No state parameter
redirect_url = f"{auth_url}?client_id={client_id}&redirect_uri={redirect_uri}"

# GOOD: Include state
state = secrets.token_urlsafe(32)
session['oauth_state'] = state
redirect_url = f"{auth_url}?client_id={client_id}&redirect_uri={redirect_uri}&state={state}"

# On callback
if request.args['state'] != session['oauth_state']:
    raise SecurityError("Invalid state parameter")
```

### 2. Open Redirect Vulnerability

```python
# BAD: Accept any redirect_uri
redirect_uri = request.args['redirect_uri']

# GOOD: Validate against whitelist
ALLOWED_REDIRECTS = [
    "https://myapp.com/callback",
    "https://staging.myapp.com/callback"
]

if redirect_uri not in ALLOWED_REDIRECTS:
    raise SecurityError("Invalid redirect_uri")
```

### 3. Token Leakage via Referrer

```html
<!-- BAD: Token in URL can leak via Referrer header -->
<a href="https://external-site.com">Click here</a>

<!-- GOOD: Use Referrer-Policy header -->
<meta name="referrer" content="no-referrer">
```

### 4. Insufficient Scope Validation

```python
# BAD: Trust token without checking scope
def delete_photo(request):
    token = validate_token(request)
    # Missing scope check!
    delete_from_db(request.photo_id)

# GOOD: Verify required scope
def delete_photo(request):
    token = validate_token(request)
    if 'photos.delete' not in token['scope'].split():
        raise ForbiddenError("Insufficient scope")
    delete_from_db(request.photo_id)
```

---

## Discovery and Metadata

### OpenID Connect Discovery

```
GET https://auth.example.com/.well-known/openid-configuration

{
    "issuer": "https://auth.example.com",
    "authorization_endpoint": "https://auth.example.com/authorize",
    "token_endpoint": "https://auth.example.com/token",
    "userinfo_endpoint": "https://auth.example.com/userinfo",
    "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
    "scopes_supported": ["openid", "profile", "email"],
    "response_types_supported": ["code", "token", "id_token"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"]
}
```

### JWKS (JSON Web Key Set)

```
GET https://auth.example.com/.well-known/jwks.json

{
    "keys": [
        {
            "kty": "RSA",
            "kid": "key-id-1",
            "use": "sig",
            "alg": "RS256",
            "n": "0vx7agoebGc...",
            "e": "AQAB"
        }
    ]
}
```

---

## Implementation Checklist

```
Authorization Server Setup:
□ Use HTTPS everywhere
□ Validate redirect_uri against whitelist
□ Implement PKCE for public clients
□ Short-lived authorization codes (< 10 min)
□ One-time use authorization codes
□ Rotate refresh tokens on use

Client Implementation:
□ Use state parameter for CSRF protection
□ Use nonce for ID token validation
□ Store tokens securely (not localStorage)
□ Validate all tokens before use
□ Handle token expiration gracefully
□ Implement proper logout (revoke tokens)

API/Resource Server:
□ Validate access tokens on every request
□ Check token scope against required permissions
□ Verify audience claim
□ Handle expired/invalid tokens with 401
□ Log authentication failures
```

---

## References

- [RFC 6749: OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc6749)
- [RFC 7636: PKCE](https://datatracker.ietf.org/doc/html/rfc7636)
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [OAuth 2.0 Security Best Current Practice](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)
- [OWASP OAuth Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/OAuth_Cheat_Sheet.html)
