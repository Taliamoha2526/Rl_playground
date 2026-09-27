"""Passenger model.
A Passenger owns its own identity, origin/destination, lifecycle
state, and timing. It does not choose an elevator, perform physics,
or render itself.
"""

from dataclasses import dataclass
from typing import Optional
from elevator.mechanics.states import PassengerState


@dataclass
class Passenger:
    """A single passenger moving through the building.
    Attributes:
        id: Unique passenger identifier.
        origin_floor: Zero based floor index where the passenger starts.
        destination_floor: Zero based floor index the passenger wants.
        request_time: Simulation time the passenger appeared/requested.
        state: Current lifecycle state.
        assigned_elevator: ID of the elevator assigned to this passenger, if any.
        pickup_time: Simulation time the passenger boarded an elevator.
        arrival_time: Simulation time the passenger reached their destination floor.
    """

    id: int
    origin_floor: int
    destination_floor: int
    request_time: float
    state: PassengerState = PassengerState.WAITING
    assigned_elevator: Optional[int] = None
    pickup_time: Optional[float] = None
    arrival_time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.origin_floor == self.destination_floor:
            raise ValueError("Passenger destination cannot equal origin")

    def assign_elevator(self, elevator_id: int) -> None:
        """Record that an elevator has been assigned to this passenger."""
        self.assigned_elevator = elevator_id

    def board(self, simulation_time: float) -> None:
        """Transition the passenger into the elevator."""
        self.state = PassengerState.RIDING
        self.pickup_time = simulation_time

    def begin_boarding(self) -> None:
        """Mark the passenger as actively boarding."""
        self.state = PassengerState.BOARDING

    def begin_exiting(self) -> None:
        """Mark the passenger as actively exiting."""
        self.state = PassengerState.EXITING

    def arrive(self, simulation_time: float) -> None:
        """Transition the passenger to arrived at their destination."""
        self.state = PassengerState.ARRIVED
        self.arrival_time = simulation_time

    @property
    def waiting_time(self) -> Optional[float]:
        """Seconds between request and pickup, or None if not yet picked up."""
        if self.pickup_time is None:
            return None
        return self.pickup_time - self.request_time

    @property
    def travel_time(self) -> Optional[float]:
        """Seconds between pickup and arrival, or None if not yet arrived."""
        if self.pickup_time is None or self.arrival_time is None:
            return None
        return self.arrival_time - self.pickup_time

    @property
    def total_journey_time(self) -> Optional[float]:
        """Seconds between request and arrival, or None if not yet arrived."""
        if self.arrival_time is None:
            return None
        return self.arrival_time - self.request_time
