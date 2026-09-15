"""Mock scheduling/insurance backend — same pattern as ai-agent-eval-harness so the two
repos read as one consistent portfolio, not two disconnected demos."""

_SLOTS = {
    "dermatology": ["2026-09-18 11:00", "2026-09-19 13:00"],
    "cardiology": ["2026-09-22 09:00", "2026-09-22 14:30", "2026-09-24 10:15"],
}
_BOOKED = []


def get_available_slots(specialty: str) -> list[str]:
    return _SLOTS.get(specialty.lower(), [])


def book_appointment(specialty: str, slot: str, patient_name: str) -> dict:
    slots = _SLOTS.get(specialty.lower(), [])
    if slot not in slots:
        return {"success": False, "reason": "slot_unavailable"}
    slots.remove(slot)
    _BOOKED.append({"specialty": specialty, "slot": slot, "patient_name": patient_name})
    return {"success": True, "confirmation_id": f"CONF-{len(_BOOKED):04d}"}


TOOL_SCHEMAS = [
    {
        "name": "get_available_slots",
        "description": "Look up open appointment slots for a given specialty.",
        "input_schema": {
            "type": "object",
            "properties": {"specialty": {"type": "string"}},
            "required": ["specialty"],
        },
    },
    {
        "name": "book_appointment",
        "description": "Book an appointment for a patient at a specific specialty and slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialty": {"type": "string"},
                "slot": {"type": "string"},
                "patient_name": {"type": "string"},
            },
            "required": ["specialty", "slot", "patient_name"],
        },
    },
]

TOOL_IMPLS = {"get_available_slots": get_available_slots, "book_appointment": book_appointment}
