"""Optional experimental conditions; absence means unknown for legacy sessions."""

from datetime import date


OPTIONAL_SESSION_FIELDS = (
    "recording_date", "electrode_placement_id", "electrode_reapplied", "arm_posture",
    "shoulder_position", "elbow_position", "forearm_orientation", "dynamic_arm_motion",
    "calibration_session_id", "notes",
)
BOOLEAN_FIELDS = ("electrode_reapplied", "dynamic_arm_motion")
CONTROLLED_VALUES = {
    "arm_posture": ("reference", "elbow_flexed", "elbow_extended", "arm_abducted",
                    "arm_near_torso", "custom", "stationary_unspecified"),
    "forearm_orientation": ("neutral", "pronated", "supinated", "custom"),
}
CONDITION_FIELDS = (
    "electrode_placement_id", "electrode_reapplied", "arm_posture", "shoulder_position",
    "elbow_position", "forearm_orientation", "dynamic_arm_motion",
)


def validate_session_conditions(values: dict) -> dict:
    """Validate supplied optional values, retaining nulls and no invented defaults."""
    if not isinstance(values, dict) or set(values) - set(OPTIONAL_SESSION_FIELDS):
        raise ValueError("condition metadata must be an object containing only optional session fields")
    validated = {}
    for key, value in values.items():
        if value is None or value == "":
            validated[key] = None
            continue
        if key in BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a JSON boolean or null")
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be text or null")
        elif key in CONTROLLED_VALUES and value not in CONTROLLED_VALUES[key]:
            raise ValueError(f"unknown {key}: {value}; use a controlled value or custom")
        elif key == "recording_date" and date.fromisoformat(value).isoformat() != value:
            raise ValueError("recording_date must be YYYY-MM-DD")
        validated[key] = value
    return validated
