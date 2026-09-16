"""TwiML builder for routing a Twilio call into the Media Streams WS endpoint. Pure string
building — no server needed to exercise this."""

from xml.sax.saxutils import escape


def stream_twiml(ws_url: str) -> str:
    """TwiML instructing Twilio to open a bidirectional Media Stream to ws_url for the duration
    of the call. Point a Twilio phone number's voice webhook at whatever serves this."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{escape(ws_url)}" /></Connect>'
        "</Response>"
    )
