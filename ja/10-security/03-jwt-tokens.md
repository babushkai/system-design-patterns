# JSON Web Token（JWT）

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../10-security/03-jwt-tokens.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

JWTはクレームをJSONとしてエンコードする自己完結型トークンであり、整合性を保証するために署名されています。ステートレス認証を可能にしますが、無効化とサイズに関するトレードオフがあります。JWTのセキュリティ問題のほとんどは、プロトコルの欠陥ではなく実装エラーに起因します。

---

## JWTの構造

JWTはドットで区切られた3つのbase64urlエンコード部分で構成されます：

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Header.Payload.Signature
```

### ヘッダー

```json
{
    "alg": "HS256",    // Signing algorithm
    "typ": "JWT"       // Token type
}
```

一般的なアルゴリズム：
- **HS256**: HMAC with SHA-256（対称鍵）
- **RS256**: RSA signature with SHA-256（非対称鍵）
- **ES256**: ECDSA with SHA-256（非対称鍵）

### ペイロード（クレーム）

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

### 署名

```
HMACSHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    secret
)
```

---

## 対称鍵 vs. 非対称鍵署名

### 対称鍵（HS256）

署名と検証に同じ秘密鍵を使用します。

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

### 非対称鍵（RS256/ES256）

秘密鍵で署名し、公開鍵で検証します。

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

### 使い分け

| シナリオ | 推奨 |
|----------|----------------|
| 単一モノリシックアプリ | HS256（よりシンプル） |
| マイクロサービス | RS256/ES256 |
| サードパーティ連携 | RS256/ES256 |
| 高セキュリティ環境 | ES256（小さく、高速） |

---

## JWTの作成

### シェルレベルでの構築

JWTを手動で構築して構造を理解します：

```bash
# 1. Create the header
HEADER=$(echo -n '{"alg":"HS256","typ":"JWT"}' | base64 | tr '+/' '-_' | tr -d '=')

# 2. Create the payload
NOW=$(date +%s)
EXP=$((NOW + 3600))
PAYLOAD=$(echo -n "{\"sub\":\"user_123\",\"iat\":$NOW,\"exp\":$EXP,\"iss\":\"my-auth-server\",\"aud\":\"my-api\",\"role\":\"admin\"}" \
  | base64 | tr '+/' '-_' | tr -d '=')

# 3. Create the signature (HS256 = HMAC-SHA256)
SIGNATURE=$(echo -n "$HEADER.$PAYLOAD" \
  | openssl dgst -sha256 -hmac "your-256-bit-secret" -binary \
  | base64 | tr '+/' '-_' | tr -d '=')

# 4. Assemble the JWT
JWT="$HEADER.$PAYLOAD.$SIGNATURE"
echo "$JWT"
```

**RS256**（非対称鍵）の場合、代わりに秘密鍵で署名します：

```bash
# Generate an RSA key pair (one-time setup)
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem

# Header for RS256
HEADER=$(echo -n '{"alg":"RS256","typ":"JWT"}' | base64 | tr '+/' '-_' | tr -d '=')

# Sign with the private key
SIGNATURE=$(echo -n "$HEADER.$PAYLOAD" \
  | openssl dgst -sha256 -sign private.pem -binary \
  | base64 | tr '+/' '-_' | tr -d '=')

JWT="$HEADER.$PAYLOAD.$SIGNATURE"
```

### Node.jsの例（jsonwebtoken）

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

## JWTの検証

### 検証チェックリスト

```bash
# 1. Decode header and payload (does NOT verify signature)
HEADER=$(echo "$JWT" | cut -d. -f1 | base64 -d 2>/dev/null)
CLAIMS=$(echo "$JWT" | cut -d. -f2 | base64 -d 2>/dev/null)

echo "$HEADER" | jq
echo "$CLAIMS" | jq

# 2. Verify algorithm is expected (reject 'none' or unexpected algorithms)
ALG=$(echo "$HEADER" | jq -r '.alg')
[ "$ALG" = "RS256" ] || { echo "Unexpected algorithm: $ALG"; exit 1; }

# 3. Verify standard claims
echo "$CLAIMS" | jq -e '.iss == "my-auth-server"'       # Issuer
echo "$CLAIMS" | jq -e '.aud == "my-api"'                # Audience
echo "$CLAIMS" | jq -e ".exp > $(date +%s)"              # Not expired
echo "$CLAIMS" | jq -e 'has("sub", "iat")'               # Required claims present

# 4. Verify signature (RS256) — fetch JWKS, then verify with openssl
curl -s https://auth.example.com/.well-known/jwks.json | jq '.keys[0]'
# Extract the public key matching the "kid" from the header, then:
echo -n "$(echo "$JWT" | cut -d. -f1-2)" \
  | openssl dgst -sha256 -verify public.pem \
    -signature <(echo "$JWT" | cut -d. -f3 | tr '_-' '/+' | base64 -d 2>/dev/null)

# 5. Additional business logic
ROLE=$(echo "$CLAIMS" | jq -r '.role')
[[ "$ROLE" == "admin" || "$ROLE" == "user" ]] || { echo "Invalid role"; exit 1; }
```

### 重要：常にアルゴリズムを指定する

```
VULNERABLE: Libraries that read "alg" from the token header and trust it.
  - Attacker sets alg=none → unsigned token accepted
  - Attacker sets alg=HS256 when server expects RS256 →
    uses public key as HMAC secret to forge tokens

SECURE: Always enforce expected algorithm on the verification side.
  - Check the header "alg" matches exactly what you expect
  - Never allow 'none'
  - Never allow both symmetric and asymmetric algorithms
```

---

## JWTのセキュリティ脆弱性

### 1. アルゴリズム混同攻撃

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

### 2. Noneアルゴリズム攻撃

```
Attack scenario:
1. Attacker sets header: {"alg": "none"}
2. Attacker removes signature
3. Poorly configured library accepts unsigned token

Prevention:
- Explicitly specify algorithms=['RS256'] in decode
- Never include 'none' in allowed algorithms
```

### 3. 弱い秘密鍵

```bash
# BAD - easily brute-forced
SECRET="secret"
SECRET="password123"

# GOOD - cryptographically random (256 bits)
SECRET=$(openssl rand -hex 32)
echo "$SECRET"
# e.g. a3f1b7c9d4e8f2...64 hex chars (32 bytes = 256 bits)
```

**ブルートフォースの現実：**

```
Secret length  | Time to crack (modern GPU)
---------------|---------------------------
8 chars        | Seconds to minutes
16 chars       | Days to weeks
32 chars       | Computationally infeasible
```

### 4. 脆弱な場所にトークンを保存

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

### 5. 有効期限なしまたは長すぎる

```bash
NOW=$(date +%s)

# BAD - No expiration
PAYLOAD='{"sub":"user123"}'

# BAD - 30-day access token
PAYLOAD="{\"sub\":\"user123\",\"exp\":$((NOW + 2592000))}"

# GOOD - Short-lived access token (15 minutes)
PAYLOAD="{\"sub\":\"user123\",\"exp\":$((NOW + 900))}"
```

---

## トークン無効化戦略

JWTはステートレスです — 設計上、無効化できません。以下は回避策です：

### 戦略1：短い有効期限 + リフレッシュトークン

```
Access Token:  15 minutes
Refresh Token: 7 days (stored in DB, revocable)

Flow:
1. User logs out
2. Delete refresh token from DB
3. Access token still valid for up to 15 min (acceptable)
4. After 15 min, refresh fails, user must re-login
```

### 戦略2：トークンブラックリスト

```bash
# Redis-based blacklist — revoke a token by its jti claim
JTI=$(echo "$JWT" | cut -d. -f2 | base64 -d 2>/dev/null | jq -r '.jti')
EXP=$(echo "$JWT" | cut -d. -f2 | base64 -d 2>/dev/null | jq -r '.exp')
TTL=$((EXP - $(date +%s)))

# Add to blacklist with TTL matching token expiration
redis-cli SETEX "blacklist:$JTI" "$TTL" "revoked"

# On every request, check if the token is blacklisted
redis-cli EXISTS "blacklist:$JTI"
# Returns 1 → token revoked, reject with 401
# Returns 0 → token not revoked, proceed
```

**トレードオフ：** すべてのリクエストにデータベース参照を追加し、ステートレスの利点を部分的に打ち消します。

### 戦略3：トークンバージョニング

```bash
# Store token version per user in DB/cache.
# When user logs out or changes password, increment the version.

# Token creation — include current version in the payload:
# {"sub":"user_123","token_version":3,"exp":...}

# Token validation — decode and compare version against DB:
TOKEN_VER=$(echo "$JWT" | cut -d. -f2 | base64 -d 2>/dev/null | jq -r '.token_version')
USER_ID=$(echo "$JWT" | cut -d. -f2 | base64 -d 2>/dev/null | jq -r '.sub')

# Fetch current version from DB/cache (e.g., Redis)
CURRENT_VER=$(redis-cli GET "user:$USER_ID:token_version")

# If they don't match, the token has been invalidated
[ "$TOKEN_VER" = "$CURRENT_VER" ] \
  && echo "Token version valid" \
  || echo "Token invalidated — return 401"
```

### 戦略4：ハイブリッドアプローチ

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

## JWTのサイズに関する考慮事項

JWTは大きくなる可能性があり、パフォーマンスに影響します。

### サイズの内訳

```
Typical JWT:
  Header:    ~36 bytes (base64)
  Payload:   ~200-500 bytes (base64)
  Signature: ~86 bytes (RS256) or ~43 bytes (HS256)
  Total:     ~300-700 bytes

Problematic JWT (too many claims):
  Payload with roles, permissions, user data: 2-4 KB
```

### サイズの影響

```
Every HTTP request includes JWT in header:
  Authorization: Bearer <token>

If token is 2KB and user makes 100 requests:
  200KB of bandwidth just for tokens

Mobile/slow networks: Significant latency impact
```

### サイズ削減戦略

```json
// BAD — embedding all user data inflates the token
{
    "sub": "user123",
    "name": "John Doe",
    "email": "john@example.com",
    "address": { "...": "..." },
    "permissions": ["read:users", "write:users", "...50 more..."],
    "roles": ["admin", "manager"]
}

// GOOD — minimal claims, fetch details when needed
{
    "sub": "user123",
    "role": "admin",
    "exp": 1704067200
}
// Fetch full permissions from cache/DB when needed
```

---

## アクセストークン vs. IDトークン

### アクセストークン

- **目的：** リソースへのアクセスを認可
- **オーディエンス：** リソースサーバー（API）
- **内容：** パーミッション、スコープ
- **検証：** リソースサーバーが検証
- **不透明性：** 不透明（非JWT）またはJWTの場合がある

### IDトークン（OpenID Connect）

- **目的：** ユーザーアイデンティティを認証
- **オーディエンス：** クライアントアプリケーション
- **内容：** ユーザーアイデンティティクレーム
- **検証：** クライアントが検証
- **形式：** 常にJWT

```bash
# Access token — for API calls to the resource server
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  https://api.example.com/data

# ID token — decode locally to get user info in the client
echo "$ID_TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | jq
# {
#   "sub": "user_12345",
#   "email": "john@example.com",
#   "name": "John Doe",
#   ...
# }
```

**重要：** IDトークンをリソースサーバーに送信しないでください。認可のためのものではありません。

---

## 実装パターン

### ミドルウェアパターン

```bash
# Validate JWT before accessing a protected endpoint
TOKEN="$1"  # Passed as argument or extracted from request

# 1. Check token is present
[ -z "$TOKEN" ] && { echo '{"error":"Missing token"}'; exit 1; }

# 2. Decode and verify claims
CLAIMS=$(echo "$TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null)
ALG=$(echo "$TOKEN" | cut -d. -f1 | base64 -d 2>/dev/null | jq -r '.alg')

[ "$ALG" = "RS256" ] || { echo '{"error":"Invalid algorithm"}'; exit 1; }
echo "$CLAIMS" | jq -e ".exp > $(date +%s)" > /dev/null 2>&1 \
  || { echo '{"error":"Token expired"}'; exit 1; }
echo "$CLAIMS" | jq -e '.aud == "my-api"' > /dev/null 2>&1 \
  || { echo '{"error":"Invalid audience"}'; exit 1; }

# 3. Call the protected resource
curl -s -H "Authorization: Bearer $TOKEN" https://api.example.com/protected
# {"user": "user_123"}
```

### スコープベースの認可

```bash
# Verify the token has the required scope before allowing access
REQUIRED_SCOPE="admin:read"

CLAIMS=$(echo "$TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null)
SCOPES=$(echo "$CLAIMS" | jq -r '.scope')

echo "$SCOPES" | tr ' ' '\n' | grep -qx "$REQUIRED_SCOPE" \
  && curl -s -H "Authorization: Bearer $TOKEN" https://api.example.com/admin \
  || echo '{"error":"Insufficient scope"}  # 403 Forbidden'
```

### Node.jsの例（Expressミドルウェア）

```javascript
const jwt = require('jsonwebtoken');

function requireAuth(req, res, next) {
    const token = (req.headers.authorization || '').replace('Bearer ', '');
    if (!token) return res.status(401).json({ error: 'Missing token' });

    try {
        req.user = jwt.verify(token, publicKey, {
            algorithms: ['RS256'],
            audience: 'my-api'
        });
        next();
    } catch (err) {
        res.status(401).json({ error: err.message });
    }
}

function requireScope(scope) {
    return [requireAuth, (req, res, next) => {
        const scopes = (req.user.scope || '').split(' ');
        if (!scopes.includes(scope)) {
            return res.status(403).json({ error: 'Insufficient scope' });
        }
        next();
    }];
}

app.get('/protected', requireAuth, (req, res) => {
    res.json({ user: req.user.sub });
});

app.get('/admin', ...requireScope('admin:read'), (req, res) => {
    res.json({ admin: true });
});
```

---

## JWTのテスト

### テストトークンの生成

```bash
# Helper: create a test JWT (HS256) with optional claim overrides
create_test_token() {
  local NOW=$(date +%s)
  local EXP=${1:-$((NOW + 3600))}
  local AUD=${2:-"test-audience"}

  local HEADER=$(echo -n '{"alg":"HS256","typ":"JWT"}' | base64 | tr '+/' '-_' | tr -d '=')
  local PAYLOAD=$(echo -n "{\"sub\":\"test_user\",\"iat\":$NOW,\"exp\":$EXP,\"iss\":\"test-issuer\",\"aud\":\"$AUD\"}" \
    | base64 | tr '+/' '-_' | tr -d '=')
  local SIG=$(echo -n "$HEADER.$PAYLOAD" \
    | openssl dgst -sha256 -hmac "test-secret" -binary \
    | base64 | tr '+/' '-_' | tr -d '=')

  echo "$HEADER.$PAYLOAD.$SIG"
}

# Test: expired token should return 401
EXPIRED_TOKEN=$(create_test_token $(($(date +%s) - 3600)))
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $EXPIRED_TOKEN" \
  https://api.example.com/protected
# Expected: 401

# Test: wrong audience should return 401
WRONG_AUD_TOKEN=$(create_test_token $(($(date +%s) + 3600)) "wrong-audience")
curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $WRONG_AUD_TOKEN" \
  https://api.example.com/protected
# Expected: 401
```

### JWTのデバッグ

```bash
# Decode JWT without verification (for debugging only!)
echo "eyJhbGciOiJIUzI1NiIs..." | cut -d. -f2 | base64 -d | jq

# Or use jwt.io (NEVER paste production tokens!)
```

---

## ベストプラクティスまとめ

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

## 参考文献

- [RFC 7519: JSON Web Token](https://datatracker.ietf.org/doc/html/rfc7519)
- [RFC 7515: JSON Web Signature](https://datatracker.ietf.org/doc/html/rfc7515)
- [RFC 7518: JSON Web Algorithms](https://datatracker.ietf.org/doc/html/rfc7518)
- [JWT Best Practices (Auth0)](https://auth0.com/blog/jwt-security-best-practices/)
- [Critical vulnerabilities in JSON Web Token libraries](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/)
