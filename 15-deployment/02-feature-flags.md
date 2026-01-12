# Feature Flags

## TL;DR

Feature flags decouple deployment from release, letting you ship code to production disabled and enable it gradually. They enable trunk-based development, safe rollouts, and experimentation. But they add complexity - plan for flag lifecycle management and cleanup.

---

## Why Feature Flags?

### Deployment vs. Release

```
Traditional:
Deploy = Release = Risk

With Feature Flags:
┌─────────────────────────────────────────────────────────────────┐
│  Code deployed to production (behind flag = OFF)                 │
│                                                                 │
│  Time T1: Flag OFF → Feature invisible to users                 │
│  Time T2: Flag ON for 1% → Test with small group               │
│  Time T3: Flag ON for 10% → Expand gradually                   │
│  Time T4: Flag ON for 100% → Full release                      │
│  Time T5: Remove flag → Clean up code                          │
│                                                                 │
│  At any point: Turn flag OFF → Instant rollback                 │
└─────────────────────────────────────────────────────────────────┘

Separation:
- Deploy: Code goes to production (low risk)
- Release: Feature enabled for users (controlled)
- Rollback: Toggle flag (instant, no redeploy)
```

---

## Types of Feature Flags

### Release Flags (Short-lived)

```python
# Gradually roll out new feature
if feature_flags.is_enabled("new_checkout_flow", user_id):
    return new_checkout_flow(cart)
else:
    return old_checkout_flow(cart)

# Lifecycle:
# 1. Deploy with flag OFF
# 2. Enable for internal users
# 3. Enable for 1%, 10%, 50%, 100%
# 4. Remove flag, delete old code
```

### Experiment Flags (Temporary)

```python
# A/B test different variants
variant = feature_flags.get_variant("checkout_button_color", user_id)

if variant == "control":
    button_color = "blue"
elif variant == "variant_a":
    button_color = "green"
elif variant == "variant_b":
    button_color = "red"

# Track conversion
analytics.track("checkout_completed", {
    "experiment": "checkout_button_color",
    "variant": variant
})

# Lifecycle:
# 1. Run experiment for statistical significance
# 2. Analyze results
# 3. Pick winner, remove flag
```

### Ops Flags (Long-lived)

```python
# Circuit breaker / kill switch
if feature_flags.is_enabled("enable_recommendations_service"):
    recommendations = recommendations_service.get(user_id)
else:
    recommendations = []  # Graceful degradation

# Lifecycle: Long-lived, used for operational control
```

### Permission Flags (Long-lived)

```python
# Entitlements / premium features
if feature_flags.is_enabled("premium_analytics", user_id):
    show_advanced_analytics()
else:
    show_upgrade_prompt()

# Lifecycle: Long-lived, tied to business logic
```

---

## Flag Evaluation

### Simple Boolean

```python
class SimpleFlag:
    def __init__(self, name: str, enabled: bool):
        self.name = name
        self.enabled = enabled
    
    def is_enabled(self) -> bool:
        return self.enabled
```

### Percentage Rollout

```python
import hashlib

class PercentageFlag:
    def __init__(self, name: str, percentage: int):
        self.name = name
        self.percentage = percentage  # 0-100
    
    def is_enabled(self, user_id: str) -> bool:
        # Consistent hashing: same user always gets same result
        hash_input = f"{self.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        return bucket < self.percentage

# 10% rollout
flag = PercentageFlag("new_feature", 10)
flag.is_enabled("user_123")  # True or False (consistent for this user)
```

### Targeting Rules

```python
class TargetedFlag:
    def __init__(self, name: str, rules: list):
        self.name = name
        self.rules = rules  # Evaluated in order
    
    def is_enabled(self, context: dict) -> bool:
        for rule in self.rules:
            if rule.matches(context):
                return rule.result
        return False  # Default

# Example rules
rules = [
    # Rule 1: Internal users always on
    Rule(
        condition=lambda ctx: ctx.get("email", "").endswith("@company.com"),
        result=True
    ),
    # Rule 2: Beta users
    Rule(
        condition=lambda ctx: ctx.get("user_id") in beta_user_list,
        result=True
    ),
    # Rule 3: 10% of remaining users
    Rule(
        condition=lambda ctx: percentage_check(ctx.get("user_id"), 10),
        result=True
    ),
    # Rule 4: Default off
    Rule(
        condition=lambda ctx: True,
        result=False
    )
]
```

### Multivariate Flags

```python
class MultivariateFlag:
    def __init__(self, name: str, variants: list):
        self.name = name
        self.variants = variants  # [("control", 50), ("variant_a", 25), ("variant_b", 25)]
    
    def get_variant(self, user_id: str) -> str:
        hash_input = f"{self.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        cumulative = 0
        for variant_name, percentage in self.variants:
            cumulative += percentage
            if bucket < cumulative:
                return variant_name
        
        return self.variants[0][0]  # Default to first
```

---

## Implementation Architecture

### Client-Side Evaluation

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Feature Flag SDK                      │   │
│  │                                                          │   │
│  │  ┌──────────────┐    ┌──────────────────────────────┐   │   │
│  │  │    Cache     │    │    Evaluation Engine         │   │   │
│  │  │              │    │                              │   │   │
│  │  │  All flag    │───►│  1. Load flag config         │   │   │
│  │  │  configs     │    │  2. Evaluate rules           │   │   │
│  │  │              │    │  3. Return result            │   │   │
│  │  └──────────────┘    └──────────────────────────────┘   │   │
│  │         ▲                                                │   │
│  │         │ Sync (polling or streaming)                    │   │
│  └─────────┼────────────────────────────────────────────────┘   │
│            │                                                     │
└────────────┼─────────────────────────────────────────────────────┘
             │
             │
┌────────────▼─────────────────────────────────────────────────────┐
│                    Feature Flag Service                          │
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │   Dashboard   │    │     API       │    │   Database    │   │
│  │               │    │               │    │               │   │
│  │  - Create     │───►│  - CRUD flags │───►│  - Flag       │   │
│  │  - Edit       │    │  - Stream     │    │    configs    │   │
│  │  - Toggle     │    │    updates    │    │  - Audit log  │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘

Pros: Low latency, works offline
Cons: All flags sent to client (size), sync delay
```

### Server-Side Evaluation

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Feature Flag SDK                      │   │
│  │                                                          │   │
│  │  is_enabled("feature_x", user_context)                  │   │
│  │              │                                           │   │
│  │              │ HTTP/gRPC                                 │   │
│  │              ▼                                           │   │
│  └──────────────┼───────────────────────────────────────────┘   │
│                 │                                                │
└─────────────────┼────────────────────────────────────────────────┘
                  │
                  │
┌─────────────────▼────────────────────────────────────────────────┐
│                    Feature Flag Service                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Evaluation Engine                         │  │
│  │                                                            │  │
│  │  1. Receive context (user_id, attributes)                 │  │
│  │  2. Load flag config                                       │  │
│  │  3. Evaluate rules                                         │  │
│  │  4. Return result                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Pros: Sensitive rules stay server-side, always fresh
Cons: Latency, network dependency
```

### Hybrid Approach

```python
class HybridFeatureFlags:
    def __init__(self):
        self.local_cache = {}
        self.evaluation_service = FeatureFlagService()
    
    def is_enabled(self, flag_name: str, context: dict) -> bool:
        # Check local cache first
        cached = self.local_cache.get(flag_name)
        if cached and not cached.requires_server_evaluation:
            return cached.evaluate(context)
        
        # Fall back to server for complex rules
        return self.evaluation_service.evaluate(flag_name, context)
    
    def sync_flags(self):
        """Background sync of flags that can be evaluated locally"""
        simple_flags = self.evaluation_service.get_all_simple_flags()
        self.local_cache.update(simple_flags)
```

---

## SDK Implementation

### Python SDK Example

```python
import requests
import threading
import time
from typing import Optional, Dict, Any

class FeatureFlagClient:
    def __init__(self, sdk_key: str, base_url: str = "https://flags.example.com"):
        self.sdk_key = sdk_key
        self.base_url = base_url
        self.flags: Dict[str, Any] = {}
        self._start_polling()
    
    def _start_polling(self):
        def poll():
            while True:
                try:
                    self._fetch_flags()
                except Exception as e:
                    print(f"Failed to fetch flags: {e}")
                time.sleep(30)  # Poll every 30 seconds
        
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
    
    def _fetch_flags(self):
        response = requests.get(
            f"{self.base_url}/api/flags",
            headers={"Authorization": f"Bearer {self.sdk_key}"}
        )
        response.raise_for_status()
        self.flags = response.json()
    
    def is_enabled(
        self, 
        flag_key: str, 
        user_id: Optional[str] = None,
        attributes: Optional[Dict] = None,
        default: bool = False
    ) -> bool:
        flag = self.flags.get(flag_key)
        if not flag:
            return default
        
        return self._evaluate(flag, user_id, attributes or {})
    
    def _evaluate(self, flag: dict, user_id: str, attributes: dict) -> bool:
        if not flag.get("enabled"):
            return False
        
        # Check targeting rules
        for rule in flag.get("rules", []):
            if self._matches_rule(rule, user_id, attributes):
                return rule.get("result", False)
        
        # Percentage rollout
        if percentage := flag.get("percentage"):
            return self._percentage_check(flag["key"], user_id, percentage)
        
        return flag.get("default", False)

# Usage
flags = FeatureFlagClient(sdk_key="sdk-key-123")

if flags.is_enabled("new_checkout", user_id="user_123"):
    show_new_checkout()
else:
    show_old_checkout()
```

---

## Best Practices

### Flag Naming Conventions

```python
# Good names - descriptive, consistent
"enable_new_checkout_flow"
"experiment_homepage_hero_variant"
"ops_circuit_breaker_recommendations"
"permission_premium_analytics"

# Bad names
"flag1"
"test"
"johns_feature"
"temporary_fix_delete_later"  # It won't be deleted
```

### Flag Lifecycle Management

```python
class FlagLifecycle:
    """Track flag status and enforce cleanup"""
    
    STATES = ["planning", "development", "testing", "rollout", "complete", "cleanup"]
    
    def __init__(self, flag_name: str):
        self.flag_name = flag_name
        self.state = "planning"
        self.created_at = datetime.now()
        self.owner = None
        self.cleanup_deadline = None
    
    def transition(self, new_state: str):
        if new_state == "rollout":
            # Set cleanup deadline when rollout starts
            self.cleanup_deadline = datetime.now() + timedelta(days=30)
        self.state = new_state
    
    def is_overdue_for_cleanup(self) -> bool:
        if self.state in ["complete", "cleanup"]:
            return datetime.now() > self.cleanup_deadline
        return False

# Automated cleanup reminders
def send_cleanup_reminders():
    for flag in get_all_flags():
        if flag.is_overdue_for_cleanup():
            send_reminder(
                to=flag.owner,
                subject=f"Feature flag '{flag.flag_name}' needs cleanup",
                body=f"Flag has been at 100% for over 30 days. Please remove."
            )
```

### Avoid Flag Debt

```python
# BAD: Nested flags (hard to reason about)
if flags.is_enabled("feature_a"):
    if flags.is_enabled("feature_b"):
        if flags.is_enabled("feature_c"):
            do_something()

# BETTER: Single flag with clear intent
if flags.is_enabled("feature_abc_combined"):
    do_something()

# BAD: Flag in shared code (affects everything)
def get_price(product):
    price = product.base_price
    if flags.is_enabled("new_pricing"):  # Too broad!
        price = calculate_new_price(product)
    return price

# BETTER: Specific scope
def get_price(product, context):
    if context.feature == "checkout" and flags.is_enabled("new_pricing_checkout"):
        return calculate_new_price(product)
    return product.base_price
```

### Testing with Flags

```python
import pytest
from unittest.mock import patch

class TestCheckoutWithFlags:
    def test_new_checkout_enabled(self):
        with patch('app.flags.is_enabled', return_value=True):
            result = process_checkout(cart)
            assert result.used_new_flow == True
    
    def test_new_checkout_disabled(self):
        with patch('app.flags.is_enabled', return_value=False):
            result = process_checkout(cart)
            assert result.used_new_flow == False
    
    def test_both_flows_produce_same_result(self):
        """Ensure new and old flow are functionally equivalent"""
        cart = create_test_cart()
        
        with patch('app.flags.is_enabled', return_value=False):
            old_result = process_checkout(cart)
        
        with patch('app.flags.is_enabled', return_value=True):
            new_result = process_checkout(cart)
        
        assert old_result.total == new_result.total
        assert old_result.items == new_result.items
```

---

## Feature Flag Services

### LaunchDarkly

```python
import ldclient
from ldclient.config import Config

ldclient.set_config(Config("sdk-key-123"))
client = ldclient.get()

user = {
    "key": "user-123",
    "email": "user@example.com",
    "custom": {
        "plan": "premium",
        "country": "US"
    }
}

# Boolean flag
show_feature = client.variation("new-feature", user, False)

# Multivariate flag
button_color = client.variation("button-color", user, "blue")
```

### Unleash (Open Source)

```python
from UnleashClient import UnleashClient

client = UnleashClient(
    url="https://unleash.example.com/api",
    app_name="my-app",
    custom_headers={"Authorization": "token"}
)
client.initialize_client()

# Check flag
if client.is_enabled("new-feature", context={"userId": "123"}):
    show_new_feature()

# With fallback
enabled = client.is_enabled("new-feature", fallback_function=lambda: False)
```

### Build vs. Buy

```
Build your own when:
- Simple use case (boolean flags only)
- Privacy/compliance requirements
- Tight budget
- Learning/control priority

Buy when:
- Need advanced targeting
- A/B testing built-in
- Multiple environments
- Audit/compliance features
- SDKs for many languages
- Don't want to maintain infrastructure

Popular options:
- LaunchDarkly (enterprise)
- Split.io (experimentation focus)
- Unleash (open source)
- Flagsmith (open source)
- ConfigCat (simple, affordable)
```

---

## Anti-Patterns

```
1. Permanent "temporary" flags
   - Set cleanup deadlines
   - Alert on stale flags
   
2. Flags that affect data models
   - Hard to roll back
   - Consider data migration instead

3. Too many flags
   - Cognitive overhead
   - Interaction complexity
   - Set org-wide limits

4. Testing only happy path
   - Test both flag states
   - Test flag transitions

5. No monitoring
   - Track flag evaluations
   - Alert on unexpected states
```

---

## References

- [Feature Toggles - Martin Fowler](https://martinfowler.com/articles/feature-toggles.html)
- [LaunchDarkly Documentation](https://docs.launchdarkly.com/)
- [Unleash Documentation](https://docs.getunleash.io/)
- [Testing Feature Flags](https://launchdarkly.com/blog/testing-with-feature-flags/)
