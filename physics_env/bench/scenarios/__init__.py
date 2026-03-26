"""Bench scenario registry."""

from .settle import SettleScenario


SCENARIOS = {
    "settle": SettleScenario(),
}
