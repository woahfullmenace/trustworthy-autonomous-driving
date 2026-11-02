"""Versioned synthetic action policy, independent of image processing."""
from src.concepts import CONCEPTS

ACTION_VERSION = "actions-v1.0"
POLICY = {
    "action_version": ACTION_VERSION,
    "concept_version": "concepts-v1.0",
    "eligibility_version": "eligibility-v1.0",
    "stop_if": {"obstacle_close": 1},
    "slow_if_any": {"pedestrian_present": 1, "cyclist_present": 1, "road_clear": 0},
    "otherwise": "GO",
}


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError("Action configuration must match the frozen actions-v1.0 policy; version changes require implementation and audit")


def concepts_to_action(concepts):
    """Return (action, readable reason) for five valid integer binary concepts."""
    for name in CONCEPTS:
        value = concepts.get(name)
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"Expected valid binary integer concept: {name}")
    if concepts["obstacle_close"]:
        return "STOP", "obstacle_close=1"
    triggers = [f"{name}={value}" for name, value in POLICY["slow_if_any"].items()
                if concepts[name] == value]
    if triggers:
        return "SLOW", "; ".join(triggers)
    return "GO", "obstacle_close=0; pedestrian_present=0; cyclist_present=0; road_clear=1"
