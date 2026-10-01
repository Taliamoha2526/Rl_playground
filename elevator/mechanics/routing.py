"""Route bookkeeping for the Capacity Aware dispatcher.
A Stop is one floor an elevator will visit as part of its current
multi passenger route, along with who boards and who alights there.
Uses the classic scan/look elevator scheduling idea:
keep serving requests in the direction you're already going before you reverse.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Stop:
    """One floor an elevator will stop at as part of its active route.
    Attributes:
        floor: Zero based floor index.
        boarding: Passenger ids who should board here.
        alighting: Passenger ids who should exit here.
    """

    floor: int
    boarding: list[int] = field(default_factory=list)
    alighting: list[int] = field(default_factory=list)


def count_pending_boardings(route: list[Stop]) -> int:
    """Total passengers this route still needs to pick up (not yet onboard)."""
    return sum(len(stop.boarding) for stop in route)


def infer_committed_direction(elevator, route: list[Stop]) -> Optional[int]:
    """Infer the direction (+1 up / -1 down) an elevator's route commits it to.
    Returns:
        1 if the route commits the elevator to moving up,
        -1 for down,
        None if there's no route or no discernible direction yet."""

    if not route:
        return None

    tolerance = elevator.config.position_tolerance
    current_pos = elevator.position_y

    for stop in route:
        stop_pos = stop.floor * elevator.config.floor_height
        if stop_pos > current_pos + tolerance:
            return 1
        if stop_pos < current_pos - tolerance:
            return -1
    return None