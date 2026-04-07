"""Bench scenario registry."""

from .air_spin import AirSpinScenario
from .drop_flat import DropFlatScenario
from .drop_tilted import DropTiltedScenario
from .front_legs_lifted import FrontLegsLiftedScenario
from .gait_smoke import GaitSmokeScenario
from .settle import SettleScenario
from .single_leg_sweep import SingleLegSweepScenario
from .slide_x import SlideXScenario


SCENARIOS = {
    "air_spin": AirSpinScenario(),
    "drop_flat": DropFlatScenario(),
    "drop_tilted": DropTiltedScenario(),
    "front_legs_lifted": FrontLegsLiftedScenario(),
    "gait_smoke": GaitSmokeScenario(),
    "settle": SettleScenario(),
    "single_leg_sweep": SingleLegSweepScenario(),
    "slide_x": SlideXScenario(),
}


def scenario_descriptions():
    return {
        name: {
            "category": scenario.category,
            "description": scenario.description,
        }
        for name, scenario in SCENARIOS.items()
    }
