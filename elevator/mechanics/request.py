"""Elevator service request model.
An ElevatorRequest represents a passenger's demand for service,
separate from the Passenger object itself.
"""

from dataclasses import dataclass
from typing import Optional
from elevator.mechanics.states import RequestStatus

@dataclass
class ElevatorRequest:
    """A request for elevator service originating from a passenger.
    Attributes:
        passenger_id: ID of the passenger who made this request.
        origin_floor: Zero based floor index of the pickup.
        destination_floor: Zero based floor index of the drop off.
        request_time: Simulation time the request was created.
        status: Current lifecycle status of the request.
        assigned_elevator: ID of the assigned elevator, if any.
    """

    passenger_id: int
    origin_floor: int
    destination_floor: int
    request_time: float
    status: RequestStatus = RequestStatus.WAITING
    assigned_elevator: Optional[int] = None

    def assign(self, elevator_id: int) -> None:
        """Mark this request as assigned to a specific elevator."""
        self.assigned_elevator = elevator_id
        self.status = RequestStatus.ASSIGNED

    def mark_picked_up(self) -> None:
        """Mark this request as fulfilled at the pickup stage."""
        self.status = RequestStatus.PICKED_UP

    def mark_completed(self) -> None:
        """Mark this request as fully completed (passenger dropped off)."""
        self.status = RequestStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        """True if this request is still waiting for an assignment."""
        return self.status == RequestStatus.WAITING
