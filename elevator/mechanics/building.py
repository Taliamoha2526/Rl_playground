"""Building model: the physical world container.
The building holds elevators and tracks passengers waiting on each floor.
"""

from dataclasses import dataclass, field
from elevator.settings_configs.config import SimulationConfig
from elevator.mechanics.elevator_module import Elevator
from elevator.mechanics.passenger import Passenger


@dataclass
class Building:
    """The physical building: floors and elevators.
    Attributes:
        config: Shared simulation configuration.
        elevators: All elevators in the building.
        waiting_passengers: Passengers currently waiting, keyed by the floor they are waiting on.
    """

    config: SimulationConfig
    elevators: list[Elevator] = field(default_factory=list)
    waiting_passengers: dict[int, list[Passenger]] = field(default_factory=dict)

    @classmethod
    def create(cls, config: SimulationConfig) -> "Building":
        """Construct a building with freshly initialized elevators."""
        elevators = [Elevator(id=i, config=config) for i in range(config.num_elevators)]
        waiting = {floor: [] for floor in range(config.num_floors)}
        return cls(config=config, elevators=elevators, waiting_passengers=waiting)

    @property
    def num_floors(self) -> int:
        return self.config.num_floors

    @property
    def floor_height(self) -> float:
        return self.config.floor_height

    def add_waiting_passenger(self, passenger: Passenger) -> None:
        """Register a passenger as waiting on their origin floor."""
        self.waiting_passengers.setdefault(passenger.origin_floor, []).append(passenger)

    def remove_waiting_passenger(self, passenger: Passenger) -> None:
        """Remove a passenger from the waiting list once they board."""
        floor_list = self.waiting_passengers.get(passenger.origin_floor, [])
        if passenger in floor_list:
            floor_list.remove(passenger)

    def waiting_at(self, floor_index: int) -> list[Passenger]:
        """All passengers currently waiting on the given floor."""
        return self.waiting_passengers.get(floor_index, [])

    def all_waiting_passengers(self) -> list[Passenger]:
        """Flat list of every currently waiting passenger."""
        result: list[Passenger] = []
        for floor_list in self.waiting_passengers.values():
            result.extend(floor_list)
        return result

    def reset(self) -> None:
        """Reset all elevators and clear waiting passengers."""
        for elevator in self.elevators:
            elevator.reset()
        self.waiting_passengers = {floor: [] for floor in range(self.config.num_floors)}
