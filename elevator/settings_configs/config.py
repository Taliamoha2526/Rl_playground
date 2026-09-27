"""Centralized tunable simulation configuration."""

from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration for a single simulation run.
    Attributes:
        num_floors: Number of floors in the building (>= 2).
        floor_height: Vertical distance between floors, in meters.
        num_elevators: Number of elevators in the building.
        elevator_capacity: Max passengers an elevator can carry.
        max_elevator_speed: Maximum elevator speed, in meters/second.
        elevator_acceleration: Acceleration magnitude, in m/s^2.
        elevator_deceleration: Deceleration magnitude, in m/s^2.
        door_open_time: Seconds for doors to go from closed to open.
        door_hold_time: Seconds doors stay open for boarding/exiting.
        door_close_time: Seconds for doors to go from open to closed.
        passenger_arrival_rate: Expected passenger arrivals per second (Poisson process rate lambda).
        render_fps: Target renderer frame rate.
        default_time_scale: Initial simulation time multiplier.
        position_tolerance: Distance (m) within which an elevator is considered at its target floor.
        velocity_epsilon: Speed (m/s) below which velocity is treated as effectively zero / idle direction.
        random_seed: Default seed for reproducible traffic generation.
    """

    num_floors: int = 10
    floor_height: float = 3.0

    num_elevators: int = 3
    elevator_capacity: int = 6

    max_elevator_speed: float = 2.5
    elevator_acceleration: float = 1.0
    elevator_deceleration: float = 1.0

    door_open_time: float = 0.5
    door_hold_time: float = 2.0
    door_close_time: float = 0.5

    passenger_arrival_rate: float = 0.10

    render_fps: int = 60
    default_time_scale: float = 1.0

    position_tolerance: float = 0.01
    velocity_epsilon: float = 0.01

    random_seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if any configuration value is invalid."""
        if self.num_floors < 2:
            raise ValueError("num_floors must be at least 2")
        if self.floor_height <= 0:
            raise ValueError("floor_height must be positive")
        if self.num_elevators < 1:
            raise ValueError("num_elevators must be positive")
        if self.elevator_capacity < 1:
            raise ValueError("elevator_capacity must be positive")
        if self.max_elevator_speed <= 0:
            raise ValueError("max_elevator_speed must be positive")
        if self.elevator_acceleration <= 0:
            raise ValueError("elevator_acceleration must be positive")
        if self.elevator_deceleration <= 0:
            raise ValueError("elevator_deceleration must be positive")
        if self.door_open_time < 0 or self.door_hold_time < 0 or self.door_close_time < 0:
            raise ValueError("door timing values must be non-negative")
        if self.passenger_arrival_rate < 0:
            raise ValueError("passenger_arrival_rate must be non-negative")
        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive")
        if self.default_time_scale <= 0:
            raise ValueError("default_time_scale must be positive")

    @property
    def building_height(self) -> float:
        """Total physical height of the building, in meters."""
        return (self.num_floors - 1) * self.floor_height

    def floor_to_position(self, floor_index: int) -> float:
        """Convert a zero based floor index to a physical y position."""
        return floor_index * self.floor_height
