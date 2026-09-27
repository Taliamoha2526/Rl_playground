"""Simulation statistics.
Derives aggregate metrics from simulation state (passengers, requests);
independent of the visual/rendering state.
"""

from dataclasses import dataclass, field
from elevator.mechanics.passenger import Passenger


@dataclass
class Statistics:
    """Tracks running and derived statistics about the simulation.
    Attributes:
        total_passengers_generated: Count of all passengers ever created.
        total_passengers_completed: Count of passengers who reached their destination.
        total_accumulated_waiting_time: Sum of waiting_time across all passengers who have been picked up.
        total_accumulated_journey_time: Sum of total_journey_time across all completed passengers.
        max_waiting_time: Largest waiting_time observed so far.
    """

    total_passengers_generated: int = 0
    total_passengers_completed: int = 0
    total_accumulated_waiting_time: float = 0.0
    total_accumulated_journey_time: float = 0.0
    max_waiting_time: float = 0.0

    _waiting_time_samples: int = field(default=0, repr=False)

    def record_passenger_created(self) -> None:
        self.total_passengers_generated += 1

    def record_pickup(self, passenger: Passenger) -> None:
        """Record a passenger's waiting time once they board."""
        waiting_time = passenger.waiting_time
        if waiting_time is None:
            return
        self.total_accumulated_waiting_time += waiting_time
        self._waiting_time_samples += 1
        self.max_waiting_time = max(self.max_waiting_time, waiting_time)

    def record_completion(self, passenger: Passenger) -> None:
        """Record a passenger's journey completion."""
        self.total_passengers_completed += 1
        journey_time = passenger.total_journey_time
        if journey_time is not None:
            self.total_accumulated_journey_time += journey_time

    @property
    def average_waiting_time(self) -> float:
        if self._waiting_time_samples == 0:
            return 0.0
        return self.total_accumulated_waiting_time / self._waiting_time_samples

    @property
    def average_journey_time(self) -> float:
        if self.total_passengers_completed == 0:
            return 0.0
        return self.total_accumulated_journey_time / self.total_passengers_completed

    def snapshot(self, waiting_count: int, inside_count: int, simulation_time: float) -> dict:
        """Return a dictionary snapshot suitable for display or logging."""
        return {"simulation_time": simulation_time, "waiting_passengers": waiting_count,
            "passengers_inside": inside_count, "completed_passengers": self.total_passengers_completed,
            "average_waiting_time": self.average_waiting_time, "max_waiting_time": self.max_waiting_time,
            "average_journey_time": self.average_journey_time, "total_passengers_generated": self.total_passengers_generated}

    def reset(self) -> None:
        self.total_passengers_generated = 0
        self.total_passengers_completed = 0
        self.total_accumulated_waiting_time = 0.0
        self.total_accumulated_journey_time = 0.0
        self.max_waiting_time = 0.0
        self._waiting_time_samples = 0
