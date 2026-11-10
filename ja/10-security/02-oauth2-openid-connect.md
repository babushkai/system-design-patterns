# OAuth 2.0とOpenID Connect

> **翻訳の状態:** この日本語版は英語版の以前の版に基づいており、現行の[英語版](../../10-security/02-oauth2-openid-connect.md)とは構成や内容が異なります。最新の内容は英語版を参照してください。

## TL;DR

OAuth 2.0は**認可**（リソースへのアクセス許可）のためのプロトコルです。OpenID Connect（OIDC）はOAuth 2.0の上に**認証**（アイデンティティの検証）を追加します。これらを組み合わせることで、パスワードを共有せずに安全な委任アクセスが実現できます。

---

## OAuthが解決する問題

OAuth以前は、サードパーティアプリにデータへのアクセスを許可するにはパスワードの共有が必要でした：

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

OAuthの解決策：クレデンシャルを共有せずに、制限付きで取り消し可能なアクセスを委任します。

---

## OAuth 2.0のロール

```mermaid
graph TD
    RO[Resource Owner<br/>You - the human user] -->|Grants permission| Client["Client<br/>(Photo Print)"]
    AuthServer["Authorization Server<br/>(Google OAuth)"] -->|Issues token| Client
    Client -->|Access Token| RS["Resource Server<br/>(Google Photos)"]
```

| ロール | 説明 | 例 |
|------|-------------|---------|
| リソースオーナー | アクセスを許可できるエンティティ | ユーザー |
| クライアント | アクセスを要求するアプリ | 写真印刷アプリ |
| リソースサーバー | 保護されたリソースをホストするAPI | Google Photos API |
| 認可サーバー | 認証後にトークンを発行 | Google OAuth |

---

## OAuth 2.0のグラントタイプ

### 認可コードグラント（Webアプリ推奨）

機密クライアントを持つサーバーサイドアプリケーションに最も安全です。

```mermaid
sequenceDiagram
    participant User
    participant Client as Client App
    participant Auth as Auth Server
    participant RS as Resource Server

    User->>Client: Click "Login with Google"
    Client-->>User: Redirect to authorization endpoint
    User->>Auth: Navigate to auth server
    Auth-->>User: Login + Consent Screen
    User->>Auth: Approve
    Auth-->>User: Redirect with authorization code
    User->>Client: Navigate back with code
    Client->>Auth: Exchange code + client_secret for tokens
    Auth-->>Client: Access Token + Refresh Token
    Client->>RS: Access API with token
    RS-->>Client: Protected Resource
```

**ステップ1：認可リクエスト**

```
GET https://auth.example.com/authorize?
    response_type=code
    &client_id=photo_print_app
    &redirect_uri=https://photoprint.com/callback
    &scope=photos.read
    &state=xyz123  # CSRF protection
```

**ステップ2：ユーザーが許可を付与**

```
Authorization Server shows:

  Photo Print App wants to:
  ☑ View your photos
  [Allow]  [Deny]
```

**ステップ3：コード付きリダイレクト**

```
HTTP 302 Found
Location: https://photoprint.com/callback?
    code=SplxlOBeZQQYbYS6WxSbIA
    &state=xyz123
```

**ステップ4：コードをトークンに交換（サーバー間通信）**

```
POST https://auth.example.com/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=SplxlOBeZQQYbYS6WxSbIA
&redirect_uri=https://photoprint.com/callback
&client_id=photo_print_app
&client_secret=secret123
```

**レスポンス：**

```json
{
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "tGzv3JOkF0XG5Qx2TlKWIA",
    "scope": "photos.read"
}
```

### 認可コード + PKCE（パブリッククライアント）

client_secretを保持できないモバイル/SPAアプリ向けです。

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

**PKCEが重要な理由：**

```mermaid
sequenceDiagram
    participant Attacker
    participant App

    Note over Attacker, App: Without PKCE (vulnerable)
    Attacker->>App: Intercept redirect (malware, URL scheme)
    Attacker->>Attacker: Exchange intercepted code for token
    Note over Attacker: ATTACKER HAS TOKEN

    Note over Attacker, App: With PKCE (secure)
    Attacker->>App: Intercept redirect
    Attacker->>Attacker: Exchange code BUT don't know code_verifier
    Note over Attacker: EXCHANGE FAILS
```

### クライアントクレデンシャルグラント（マシン間通信）

ユーザーが関与しません。サービスが直接認証します。

```
POST https://auth.example.com/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=backend_service
&client_secret=secret123
&scope=internal.read
```

**ユースケース：** マイクロサービスAがマイクロサービスBを呼び出す場合。

### リフレッシュトークングラント

ユーザーの操作なしに、リフレッシュトークンを新しいアクセストークンに交換します。

```
POST https://auth.example.com/token

grant_type=refresh_token
&refresh_token=tGzv3JOkF0XG5Qx2TlKWIA
&client_id=photo_print_app
&client_secret=secret123
```

---

## OpenID Connect（OIDC）

OIDCはOAuth 2.0の上にアイデンティティレイヤーを追加します。

### OAuth 2.0 vs. OpenID Connect

```
OAuth 2.0:
  Q: "Can this app access my photos?"
  A: "Yes, here's an access token"

OpenID Connect:
  Q: "Who is this user?"
  A: "Here's an ID token with user info"
```

### IDトークン

認証済みユーザーに関するクレームを含むJWTです。

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

**主要なクレーム：**

| クレーム | 説明 |
|-------|-------------|
| iss | 発行者（トークンの発行元） |
| sub | サブジェクト（一意のユーザー識別子） |
| aud | オーディエンス（意図された受信者） |
| exp | 有効期限 |
| iat | 発行時刻 |
| nonce | リプレイ攻撃の防止 |

### OIDCフロー

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

**標準OIDCスコープ：**

| スコープ | 返されるクレーム |
|-------|-----------------|
| openid | sub（OIDCに必須） |
| profile | name、family_name、pictureなど |
| email | email、email_verified |
| address | address |
| phone | phone_number、phone_number_verified |

### UserInfoエンドポイント

アクセストークンで追加のユーザー情報を取得します：

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

## トークン検証

### アクセストークンの検証

**オプション1：イントロスペクション（不透明トークン）**

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

**オプション2：ローカル検証（JWT）**

```bash
# 1. Fetch public keys from JWKS endpoint
curl -s https://auth.example.com/.well-known/jwks.json | jq

# Response:
# {
#   "keys": [
#     {
#       "kty": "RSA",
#       "kid": "key-id-1",
#       "use": "sig",
#       "alg": "RS256",
#       "n": "0vx7agoebGc...",
#       "e": "AQAB"
#     }
#   ]
# }

# 2. Decode token header to find which key was used
echo "$ACCESS_TOKEN" | cut -d. -f1 | base64 -d 2>/dev/null | jq
# {"alg":"RS256","kid":"key-id-1","typ":"JWT"}

# 3. Decode payload to inspect claims (does NOT verify signature)
echo "$ACCESS_TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | jq
# {
#   "sub": "user_12345",
#   "aud": "my_api",
#   "iss": "https://auth.example.com",
#   "exp": 1704067200,
#   "scope": "photos.read"
# }

# Signature verification requires a library — match "kid" from the header
# to the JWKS key, then verify RS256 signature with that public key.
```

### IDトークン検証チェックリスト

```bash
# Decode the ID token payload (does NOT verify signature)
CLAIMS=$(echo "$ID_TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null)
echo "$CLAIMS" | jq

# Validation steps to perform:
# 1. Verify issuer matches your auth server
echo "$CLAIMS" | jq -e '.iss == "https://auth.example.com"'

# 2. Verify audience is your client_id
echo "$CLAIMS" | jq -e '.aud == "photo_print_app"'

# 3. Verify token is not expired
echo "$CLAIMS" | jq -e ".exp > $(date +%s)"

# 4. Verify nonce matches (prevents replay)
echo "$CLAIMS" | jq -e ".nonce == \"$EXPECTED_NONCE\""

# 5. Verify signature — fetch JWKS and verify RS256 signature
#    using the public key matching the "kid" in the token header.
#    This step requires a crypto library (openssl, jose-cli, etc.).
```

---

## トークン保存のベストプラクティス

### Webアプリケーション

```
Access Token:
  - Store in memory (JavaScript variable)
  - NOT in localStorage (XSS vulnerable)
  - NOT in sessionStorage (XSS vulnerable)

Refresh Token:
  - Store in HttpOnly, Secure cookie
  - Or use Backend-for-Frontend (BFF) pattern
```

### Backend-for-Frontend（BFF）パターン

```mermaid
graph TD
    subgraph Browser
        SPA["SPA (React/Vue)<br/>Only has session cookie<br/>No tokens in JavaScript"]
    end
    SPA -->|Session cookie| BFF["BFF (Backend)<br/>Holds tokens<br/>Manages refresh"]
    BFF -->|Access Token| RS[Resource Server]
```

### モバイルアプリケーション

```
Refresh Token:
  - iOS: Keychain (kSecClassGenericPassword)
  - Android: EncryptedSharedPreferences or Keystore

Access Token:
  - Memory only, re-fetch on app restart
```

---

## よくあるセキュリティの落とし穴

### 1. Stateパラメータの欠落（CSRF）

```bash
# BAD: No state parameter
# GET https://auth.example.com/authorize?client_id=my_app&redirect_uri=https://myapp.com/callback

# GOOD: Generate a random state and include it
STATE=$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=')
# Store $STATE in the user's session, then redirect:
# GET https://auth.example.com/authorize?client_id=my_app&redirect_uri=https://myapp.com/callback&state=$STATE

# On callback, verify the returned state matches the stored value.
# If it does not match, reject the request (CSRF attempt).
```

### 2. オープンリダイレクトの脆弱性

```
BAD: Accept any redirect_uri from the request

GOOD: Authorization server must validate redirect_uri against a
      pre-registered whitelist before issuing a redirect.

Allowed redirect URIs (registered at client creation):
  - https://myapp.com/callback
  - https://staging.myapp.com/callback

Any request with a redirect_uri NOT in this list must be rejected
with an error — never redirect the user to an unregistered URI.
```

### 3. Referrerによるトークン漏洩

```html
<!-- BAD: Token in URL can leak via Referrer header -->
<a href="https://external-site.com">Click here</a>

<!-- GOOD: Use Referrer-Policy header -->
<meta name="referrer" content="no-referrer">
```

### 4. 不十分なスコープ検証

```bash
# BAD: Trust token without checking scope
# Server handles DELETE /photos/42 without verifying scope — dangerous.

# GOOD: Verify required scope before acting
# Decode the access token and check the scope claim:
SCOPE=$(echo "$ACCESS_TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | jq -r '.scope')

# Verify 'photos.delete' is present in the space-delimited scope string
echo "$SCOPE" | tr ' ' '\n' | grep -qx 'photos.delete' \
  && echo "Scope OK" \
  || echo "Insufficient scope — return 403 Forbidden"
```

---

## ディスカバリーとメタデータ

### OpenID Connectディスカバリー

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

### JWKS（JSON Web Key Set）

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

## 実装チェックリスト

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

## 参考文献

- [RFC 6749: OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc6749)
- [RFC 7636: PKCE](https://datatracker.ietf.org/doc/html/rfc7636)
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html)
- [OAuth 2.0 Security Best Current Practice](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)
- [OWASP OAuth Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/OAuth_Cheat_Sheet.html)
