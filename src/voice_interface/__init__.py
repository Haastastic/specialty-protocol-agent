"""Voice interface: wraps ProtocolAgent.respond() with streaming STT/TTS and real-time call
handling. See CLAUDE.md architecture principle 4 — nothing here reaches back into
src/protocol_agent/, the dependency only ever points this direction."""
