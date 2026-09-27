"""Elevator dispatch policy.
Responsible only for choosing an elevator for a request.
**Not responsible for moving elevators, rendering, or passenger generation.
"""

from abc import ABC, abstractmethod
from typing import Optional
from elevator.mechanics.elevator_module import Elevator
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

