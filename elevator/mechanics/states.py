from enum import Enum, auto


class Direction(Enum):
    """Direction of elevator travel, derived from velocity."""

    UP = auto()
    DOWN = auto()
    IDLE = auto()


class ElevatorState(Enum):
    """Top level operational state of an elevator."""

    IDLE = auto()
    MOVING = auto()
    STOPPED = auto()
    DOOR_OPENING = auto()
    DOOR_OPEN = auto()
    DOOR_CLOSING = auto()


class MotionPhase(Enum):
    """Internal phase of motion while an elevator is moving."""

    NONE = auto()
    ACCELERATING = auto()
    CRUISING = auto()
    DECELERATING = auto()


class DoorState(Enum):
    """Physical state of an elevator's doors."""

    CLOSED = auto()
    OPENING = auto()
    OPEN = auto()
    CLOSING = auto()


class EventType(Enum):
    """Kinds of events the simulation can emit."""

    PASSENGER_CREATED = auto()
    ELEVATOR_ASSIGNED = auto()
    ELEVATOR_ARRIVED = auto()
    PASSENGER_BOARDED = auto()
    PASSENGER_DROPPED_OFF = auto()


class PassengerState(Enum):
    """Lifecycle state of a passenger."""

    WAITING = auto()
    BOARDING = auto()
    RIDING = auto()
    EXITING = auto()
    ARRIVED = auto()


class RequestStatus(Enum):
    """Lifecycle state of an elevator service request."""

    WAITING = auto()
    ASSIGNED = auto()
    PICKED_UP = auto()
    COMPLETED = auto()
