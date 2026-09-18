"""Twilio request-signature validation (https://www.twilio.com/docs/usage/security#validating-requests).

Kept dependency-free (stdlib HMAC-SHA1) rather than pulling in the full `twilio` SDK for one
function. Without this, a publicly hosted /twiml or /media-stream lets anyone spend the
Deepgram/ElevenLabs/Anthropic credits behind it.
"""

import base64
import hashlib
import hmac
import os


def compute_signature(auth_token: str, url: str, params: dict[str, str] | None = None) -> str:
    """Twilio's algorithm: URL, then each POST param as key+value in sorted key order, HMAC-SHA1'd
    with the account auth token and base64-encoded. The WebSocket handshake has no params."""
    data = url + "".join(f"{k}{v}" for k, v in sorted((params or {}).items()))
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def is_valid_signature(
    auth_token: str, signature: str | None, urls: list[str], params: dict[str, str] | None = None
) -> bool:
    """True if `signature` matches any candidate URL. Several are accepted because behind a proxy
    the URL Twilio signed (public https/wss host) is reconstructed, not observed, and the
    WebSocket handshake may be signed under either its wss:// or https:// form."""
    if not signature:
        return False
    return any(hmac.compare_digest(compute_signature(auth_token, u, params), signature) for u in urls)


def signature_check_enabled() -> bool:
    """Fail closed: validation is on unless explicitly disabled for local development."""
    return os.environ.get("TWILIO_SKIP_SIGNATURE_CHECK", "").lower() not in ("1", "true", "yes")
