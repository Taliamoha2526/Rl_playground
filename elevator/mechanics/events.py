"""Events record notable occurrences in the simulation (passenger
creation, assignment, arrival, boarding, drop-off) with a simulation
timestamp and relevant data.
"""
from typing import Any
from dataclasses import dataclass, field
from elevator.mechanics.states import EventType

@dataclass(frozen=True)
class SimulationEvent:
    """A single timestamped simulation event.
    Attributes:
        simulation_time: The simulation time at which this occurred.
        event_type: The kind of event.
        data: Arbitrary event specific payload (IDs, floors, etc.).
        message: A short, human-readable description for logging.
    """

    simulation_time: float
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class EventBus:
    """Collects and distributes simulation events.
    The simulation engine emits events here;
    any number of listeners (event log, statistics, renderer) can subscribe (receive notice of the event that occurred).
    """

    def __init__(self) -> None:
        self._listeners: list = []
        self._history: list[SimulationEvent] = []

    def subscribe(self, listener) -> None:
        """Register a callable to receive every emitted event."""
        self._listeners.append(listener)

    def emit(self, event: SimulationEvent) -> None:
        """Record the event and notify all listeners."""
        self._history.append(event)
        for listener in self._listeners:
            listener(event)

    @property
    def history(self) -> list[SimulationEvent]:
        """Full chronological list of emitted events."""
        return self._history

    def recent(self, count: int) -> list[SimulationEvent]:
        """Return the most recent count events, oldest first."""
        if count <= 0:
            return []
        return self._history[-count:]

    def clear(self) -> None:
        """Remove all recorded event history (listeners are kept)."""
        self._history.clear()
