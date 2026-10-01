"""Elevator dispatch policy.
Responsible only for choosing an elevator for a request.
**Not responsible for moving elevators, rendering, or passenger generation.
"""

from abc import ABC, abstractmethod
from elevator.mechanics.elevator_module import Elevator
from elevator.mechanics.states import ElevatorState
from elevator.mechanics.routing import *
from elevator.mechanics.request import ElevatorRequest


class Dispatcher(ABC):
    """General interface for elevator dispatch policies."""
    @abstractmethod
    def assign(self, request: ElevatorRequest, elevators: list[Elevator]) -> Optional[int]:
        """Choose an elevator for the given request.
        Args:
            request: The pending request needing an assignment.
            elevators: All elevators in the building.
        Returns:
            The chosen elevator's id, or None if no elevator is currently available.
        """
        raise NotImplementedError


class NearestAvailableDispatcher(Dispatcher):
    """Assigns the nearest currently available elevator to a request."""

    def assign(self, request: ElevatorRequest, elevators: list[Elevator]) -> Optional[int]:
        available = [e for e in elevators if e.is_available]
        if not available:
            return None
        origin_position = request.origin_floor * available[0].config.floor_height
        nearest = min(available, key=lambda e: abs(e.position_y - origin_position))
        return nearest.id


class CapacityAwareDispatcher(Dispatcher):
    """A dispatch policy that actually uses elevator capacity.
    NearestAvailableDispatcher only ever looks at elevators that are completely free,
    and the base simulation engine only ever lets an elevator serve one passenger(pick up,
    drop off, then look for the next job), it's a deliberately unoptimized baseline.

    CapacityAwareDispatcher fixes that; given a new request, it first
    looks for an elevator that is already moving in a compatible direction,
    has the pickup floor still ahead of it, and has room left and adds
    the new pickup/drop off to that elevator's route instead of making the
    passenger wait for an idle car.
    """

    def assign(self, request: ElevatorRequest, elevators: list[Elevator], elevator_routes: Optional[dict[int, list[Stop]]] = None) -> Optional[int]:
        """Assigns requests to in progress routes when possible, else nearest idle car."""
        elevator_routes = elevator_routes or {}
        request_direction = 1 if request.destination_floor > request.origin_floor else -1

        merge_candidate = self._find_merge_candidate(request, request_direction, elevators, elevator_routes)
        if merge_candidate is not None:
            return merge_candidate.id

        idle_elevators = [e for e in elevators if e.is_available]
        if not idle_elevators:
            return None

        origin_position = request.origin_floor * idle_elevators[0].config.floor_height
        nearest_idle = min(idle_elevators, key=lambda e: abs(e.position_y - origin_position))
        return nearest_idle.id

    def _find_merge_candidate(self,request: ElevatorRequest, request_direction: int, elevators: list[Elevator], elevator_routes: dict[int, list[Stop]]) -> Optional[Elevator]:
        candidates = []

        for elevator in elevators:
            route = elevator_routes.get(elevator.id, [])
            if not route:
                continue  # no merging
            if (elevator.state in (ElevatorState.DOOR_OPEN, ElevatorState.DOOR_CLOSING)
                    and route[0].floor == request.origin_floor):
                continue
            committed_direction = infer_committed_direction(elevator, route)
            if committed_direction != request_direction:
                continue

            pickup_position = request.origin_floor * elevator.config.floor_height
            tolerance = elevator.config.position_tolerance
            if committed_direction == 1 and pickup_position < elevator.position_y - tolerance:
                continue  # pickup floor is behind the elevator, can't merge
            if committed_direction == -1 and pickup_position > elevator.position_y + tolerance:
                continue

            reserved = len(elevator.passengers) + count_pending_boardings(route)
            if reserved >= elevator.config.elevator_capacity:
                continue  # no room left, even conservatively

            candidates.append(elevator)

        if not candidates:
            return None

        pickup_position = request.origin_floor * candidates[0].config.floor_height
        return min(candidates, key=lambda e: abs(e.position_y - pickup_position))
