"""Agent communication: typed envelopes over a durable outbox."""

from enigma.comm.bus import MessageBus, idempotent_message_id

__all__ = ["MessageBus", "idempotent_message_id"]
