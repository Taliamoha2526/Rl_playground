"""Elevator physical model.
Responsible for: physical state, continuous kinematic motion,
braking distance based deceleration, the door lifecycle based on motion speed, and
elevator local passenger occupancy.
"""

from dataclasses import dataclass, field
from typing import Optional
from elevator.settings_configs.config import SimulationConfig
from elevator.mechanics.passenger import Passenger
from elevator.mechanics.states import DoorState, Direction, ElevatorState, MotionPhase

@dataclass
class Elevator:
    """A single elevator car with continuous physical state.
    Position is measured in meters along the building's vertical
    axis, where ground sits at position 0.0.
    Attributes:
        id: Unique elevator identifier.
        config: Shared simulation configuration.
        position_y: Current physical position, in meters.
        velocity: Current velocity, in meters/second (signed).
        direction: Derived travel direction.
        state: Top level operational state.
        motion_phase: Internal motion phase while moving.
        door_state: Current door state.
        door_progress: 0.0 (fully closed/at start of transition) to
            1.0 (fully open/at end of transition), used by the
            renderer to animate doors smoothly.
        target_floor: Zero based floor index the elevator is heading to, or None if idle.
        passengers: Passengers currently riding this elevator.
    """

    id: int
    config: SimulationConfig
    position_y: float = 0.0
    velocity: float = 0.0
    direction: Direction = Direction.IDLE
    state: ElevatorState = ElevatorState.IDLE
    motion_phase: MotionPhase = MotionPhase.NONE
    door_state: DoorState = DoorState.CLOSED
    door_progress: float = 0.0
    target_floor: Optional[int] = None
    passengers: list[Passenger] = field(default_factory=list)
    _door_timer: float = field(default=0.0, repr=False)

    @property
    def current_floor_estimate(self) -> int:
        """Nearest floor index to the elevator's current position."""
        floor = round(self.position_y / self.config.floor_height)
        return max(0, min(self.config.num_floors - 1, floor))

    @property
    def is_available(self) -> bool:
        """True if this elevator can accept a new assignment."""
        return self.state == ElevatorState.IDLE and self.target_floor is None

    @property
    def has_capacity(self) -> bool:
        """True if the elevator can hold at least one more passenger."""
        return len(self.passengers) < self.config.elevator_capacity

    def target_position(self) -> Optional[float]:
        """Physical y position of the current target floor, if any."""
        if self.target_floor is None:
            return None
        return self.config.floor_to_position(self.target_floor)

    def assign_target(self, floor_index: int) -> None:
        """Send the elevator toward a floor, starting motion.
        If already at the target floor, do not start moving; just stay stopped so doors can proceed.
        """
        self.target_floor = floor_index
        target_pos = self.config.floor_to_position(floor_index)

        if abs(target_pos - self.position_y) <= self.config.position_tolerance:
            self.state = ElevatorState.STOPPED
            self.velocity = 0.0
            self.direction = Direction.IDLE
            self.motion_phase = MotionPhase.NONE
        else:
            self.state = ElevatorState.MOVING

    def open_doors(self) -> None:
        """Begin the door opening sequence (only valid when stopped)."""
        if self.state != ElevatorState.STOPPED:
            return
        self.state = ElevatorState.DOOR_OPENING
        self.door_state = DoorState.OPENING
        self._door_timer = 0.0

    def board(self, passenger: Passenger) -> bool:
        """Add a passenger to this elevator if there is room.
        Returns:
            True if the passenger successfully boarded.
        """
        if not self.has_capacity:
            return False
        self.passengers.append(passenger)
        return True

    def remove_passenger(self, passenger: Passenger) -> None:
        """Remove a passenger from this elevator (they have exited)."""
        if passenger in self.passengers:
            self.passengers.remove(passenger)

    def update(self, dt: float) -> None:
        """Advance the elevator's physical and door state by dt seconds."""
        if dt <= 0:
            return
        if self.state == ElevatorState.MOVING:
            self._update_motion(dt)
        elif self.state in (ElevatorState.DOOR_OPENING, ElevatorState.DOOR_OPEN, ElevatorState.DOOR_CLOSING,):
            self._update_doors(dt)
        self._update_direction()

    def _update_direction(self) -> None:
        """ Update the direction it is headed based on the velocity."""
        eps = self.config.velocity_epsilon
        if self.velocity > eps:
            self.direction = Direction.UP
        elif self.velocity < -eps:
            self.direction = Direction.DOWN
        else:
            self.direction = Direction.IDLE

    def _update_motion(self, dt: float) -> None:
        """Continuous kinematic update with braking distance deceleration."""
        target_pos = self.target_position()
        if target_pos is None:
            self.state = ElevatorState.IDLE
            self.velocity = 0.0
            self.motion_phase = MotionPhase.NONE
            return

        remaining = target_pos - self.position_y
        travel_direction = 1.0 if remaining > 0 else -1.0
        distance_to_target = abs(remaining)

        speed = abs(self.velocity)
        # d_brake = v^2 / (2a)
        braking_distance = (speed ** 2) / (2 * self.config.elevator_deceleration)

        if distance_to_target <= braking_distance:
            # Decelerate toward zero as it approaches the target floor.
            new_speed = max(0.0, speed - self.config.elevator_deceleration * dt)
            self.motion_phase = MotionPhase.DECELERATING
        elif speed < self.config.max_elevator_speed:
            new_speed = min(self.config.max_elevator_speed, speed + self.config.elevator_acceleration * dt)
            self.motion_phase = MotionPhase.ACCELERATING
        else:
            new_speed = self.config.max_elevator_speed
            self.motion_phase = MotionPhase.CRUISING

        self.velocity = travel_direction * new_speed

        step = self.velocity * dt
        # Prevent overshoot: never step past the target position.
        if abs(step) >= distance_to_target:
            self.position_y = target_pos
            self.velocity = 0.0
            self.motion_phase = MotionPhase.NONE
            self.state = ElevatorState.STOPPED
        else:
            self.position_y += step

    def _update_doors(self, dt: float) -> None:
        """ Update door's movement based on the timer and config open/close times"""
        self._door_timer += dt

        if self.state == ElevatorState.DOOR_OPENING:
            duration = self.config.door_open_time
            self.door_progress = min(1.0, self._door_timer / duration) if duration > 0 else 1.0
            if self._door_timer >= duration:
                self.state = ElevatorState.DOOR_OPEN
                self.door_state = DoorState.OPEN
                self.door_progress = 1.0
                self._door_timer = 0.0

        elif self.state == ElevatorState.DOOR_OPEN:
            if self._door_timer >= self.config.door_hold_time:
                self.state = ElevatorState.DOOR_CLOSING
                self.door_state = DoorState.CLOSING
                self._door_timer = 0.0

        elif self.state == ElevatorState.DOOR_CLOSING:
            duration = self.config.door_close_time
            self.door_progress = (max(0.0, 1.0 - self._door_timer / duration) if duration > 0 else 0.0)
            if self._door_timer >= duration:
                self.door_state = DoorState.CLOSED
                self.door_progress = 0.0
                self._door_timer = 0.0
                if self.target_floor is not None and abs(self.target_position() - self.position_y) <= self.config.position_tolerance:
                    self.target_floor = None
                self.state = ElevatorState.IDLE

    def reset(self) -> None:
        """Restore the elevator to its V1 initial state (floor 1, idle)."""
        self.position_y = 0.0
        self.velocity = 0.0
        self.direction = Direction.IDLE
        self.state = ElevatorState.IDLE
        self.motion_phase = MotionPhase.NONE
        self.door_state = DoorState.CLOSED
        self.door_progress = 0.0
        self.target_floor = None
        self.passengers.clear()
        self._door_timer = 0.0
