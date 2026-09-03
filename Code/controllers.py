"""Non-learning traffic-signal controllers and topology validation helpers."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


TLS_ID = "TL"
ACTION_TO_GREEN_PHASE = {action: action * 2 for action in range(8)}

# This documented mapping is checked against environment.net.xml and
# tls.add.xml before max-pressure evaluation begins. The controller then uses
# the mapping derived from those files, rather than trusting this constant.
ACTION_INCOMING_LANES = {
    0: ("N2TL_0", "N2TL_1", "S2TL_0", "S2TL_1"),
    1: ("N2TL_2", "E2TL_0", "S2TL_2", "W2TL_0"),
    2: ("E2TL_0", "E2TL_1", "E2TL_2", "W2TL_0", "W2TL_1", "W2TL_2"),
    3: ("N2TL_0", "E2TL_3", "S2TL_0", "W2TL_3"),
    4: ("N2TL_0", "N2TL_1", "N2TL_2", "E2TL_0"),
    5: ("E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3", "S2TL_0"),
    6: ("S2TL_0", "S2TL_1", "S2TL_2", "W2TL_0"),
    7: ("W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3", "N2TL_0"),
}


def derive_action_incoming_lanes(net_file, tls_file, tls_id=TLS_ID):
    """Derive unique incoming lanes served by each admissible green phase.

    Connections are indexed by SUMO ``linkIndex``. Multiple link indices may
    share an incoming lane (for example when one turn connects to several
    downstream lanes), so the returned lane collection is explicitly a set.
    """
    net_root = ET.parse(Path(net_file)).getroot()
    tls_root = ET.parse(Path(tls_file)).getroot()

    links = {}
    for connection in net_root.findall("connection"):
        if connection.get("tl") != tls_id:
            continue
        link_index = int(connection.get("linkIndex"))
        incoming_lane = f"{connection.get('from')}_{connection.get('fromLane')}"
        links[link_index] = incoming_lane

    logic = next(
        (item for item in tls_root.findall("tlLogic") if item.get("id") == tls_id),
        None,
    )
    if logic is None:
        raise ValueError(f"Traffic-light logic {tls_id!r} is missing from {tls_file}")

    phases = logic.findall("phase")
    derived = {}
    for action, phase_index in ACTION_TO_GREEN_PHASE.items():
        try:
            state = phases[phase_index].get("state")
        except IndexError as exc:
            raise ValueError(f"Missing green phase {phase_index} for action {action}") from exc
        if len(state) <= max(links):
            raise ValueError(
                f"Phase {phase_index} state has {len(state)} signals but linkIndex "
                f"{max(links)} exists in {net_file}"
            )
        derived[action] = tuple(
            sorted({lane for index, lane in links.items() if state[index] in {"G", "g"}})
        )
    return derived


def validate_action_incoming_lanes(net_file, tls_file):
    """Validate the documented eight-action lane mapping against SUMO XML."""
    derived = derive_action_incoming_lanes(net_file, tls_file)
    expected = {action: tuple(sorted(lanes)) for action, lanes in ACTION_INCOMING_LANES.items()}
    if derived != expected:
        differences = {
            action: {"expected": expected[action], "derived": derived.get(action, ())}
            for action in expected
            if expected[action] != derived.get(action)
        }
        raise ValueError(f"Action/lane mapping does not match the SUMO topology: {differences}")
    return derived


class MaxPressureController:
    """Deterministic lane-queue max-pressure controller.

    The network has a single intersection followed by long outgoing links to
    dead-end destination nodes. Consequently, downstream queues are normally
    negligible and pressure is the sum of lane halting counts over the unique
    incoming lanes served by an action. This avoids counting an incoming queue
    once per raw SUMO link index.
    """

    configuration = "lane_halting_pressure_v1"

    def __init__(self, net_file, tls_file):
        self.action_lanes = validate_action_incoming_lanes(net_file, tls_file)
        self.last_pressures = None

    def compute_pressures(self, traci_like):
        self.last_pressures = {
            action: sum(
                traci_like.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in lane_ids
            )
            for action, lane_ids in self.action_lanes.items()
        }
        return self.last_pressures

    def choose_action(self, traci_like, current_action):
        """Choose maximum pressure, retaining the current action on a tie."""
        pressures = self.compute_pressures(traci_like)
        max_pressure = max(pressures.values())
        maximizing = [
            action for action, pressure in pressures.items() if pressure == max_pressure
        ]
        if current_action in maximizing:
            return current_action
        return min(maximizing)
