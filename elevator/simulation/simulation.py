"""Central simulation engine.
Coordinates the clock, building, traffic generator, dispatcher, requests, passengers, events, and statistics.
Does not perform detailed rendering or detailed elevator physics.
Its job is orchestration between modules and components.
"""

from collections import deque
from typing import Optional
from elevator.mechanics.building import Building
from elevator.mechanics.clock import SimulationClock
from elevator.settings_configs.config import SimulationConfig
from elevator.settings_configs.dispatcher import Dispatcher, NearestAvailableDispatcher
from elevator.mechanics.events import EventBus, EventType, SimulationEvent
from elevator.mechanics.passenger import Passenger
from elevator.mechanics.request import ElevatorRequest
from elevator.mechanics.states import ElevatorState, RequestStatus
from elevator.mechanics.statistics import Statistics
from elevator.settings_configs.traffic import PassengerGenerator, RandomTrafficGenerator

class ElevatorSimulation:
    """Top level simulation orchestrator.
    Usage:
        sim = ElevatorSimulation(config)
        sim.update(dt)
    """

    def __init__(self, config: Optional[SimulationConfig] = None, dispatcher: Optional[Dispatcher] = None,traffic_generator: Optional[PassengerGenerator] = None):
        self.config = config or SimulationConfig()
        self.clock = SimulationClock(time_scale=self.config.default_time_scale)
        self.building = Building.create(self.config)
        self.dispatcher = dispatcher or NearestAvailableDispatcher()
        self.traffic_generator = traffic_generator or RandomTrafficGenerator(self.config)
        self.events = EventBus()
        self.stats = Statistics()
        self.passengers: dict[int, Passenger] = {}
        self.requests: list[ElevatorRequest] = []
        self._pending_requests: deque[ElevatorRequest] = deque()
        self._elevator_request: dict[int, ElevatorRequest] = {}
        self._elevator_stage: dict[int, str] = {}

    def update(self, dt: float) -> None:
        """Advance the entire simulation by dt simulation seconds."""
        if dt <= 0:
            return

        self.clock.simulation_time += dt
        sim_time = self.clock.simulation_time
        self._generate_traffic(sim_time, dt)
        self._assign_pending_requests()
        self._update_elevators(dt)
        self._update_statistics()

    def step(self, real_dt: float) -> None:
        """Compute scaled sim time and update in one call."""
        if self.clock.paused or real_dt <= 0:
            return
        sim_dt = real_dt * self.clock.time_scale
        self.update(sim_dt)

    def _generate_traffic(self, sim_time: float, dt: float) -> None:
        new_passengers = self.traffic_generator.update(sim_time, dt)
        for passenger in new_passengers:
            self.passengers[passenger.id] = passenger
            self.building.add_waiting_passenger(passenger)
            self.stats.record_passenger_created()

            request = ElevatorRequest(passenger_id=passenger.id,  origin_floor=passenger.origin_floor,
                destination_floor=passenger.destination_floor, request_time=sim_time)
            self.requests.append(request)
            self._pending_requests.append(request)

            self.events.emit(SimulationEvent(simulation_time=sim_time, event_type=EventType.PASSENGER_CREATED,
                    data={"passenger_id": passenger.id, "origin_floor": passenger.origin_floor, "destination_floor": passenger.destination_floor,},
                    message=(f"Passenger {passenger.id} requested " f"F{passenger.origin_floor + 1} -> F{passenger.destination_floor + 1}")))

    def _assign_pending_requests(self) -> None:
        still_pending: deque[ElevatorRequest] = deque()

        while self._pending_requests:
            request = self._pending_requests.popleft()
            if request.status != RequestStatus.WAITING:
                continue

            elevator_id = self.dispatcher.assign(request, self.building.elevators)
            if elevator_id is None:
                still_pending.append(request)
                continue

            self._start_trip(elevator_id, request)

        self._pending_requests = still_pending

    def _start_trip(self, elevator_id: int, request: ElevatorRequest) -> None:
        elevator = self._get_elevator(elevator_id)
        request.assign(elevator_id)
        self._elevator_request[elevator_id] = request
        self._elevator_stage[elevator_id] = "to_pickup"
        elevator.assign_target(request.origin_floor)

        self.events.emit(SimulationEvent(simulation_time=self.clock.simulation_time, event_type=EventType.ELEVATOR_ASSIGNED,
                data={"elevator_id": elevator_id, "passenger_id": request.passenger_id},
                message=f"Elevator {elevator_id} assigned to passenger {request.passenger_id}"))

    def _update_elevators(self, dt: float) -> None:
        for elevator in self.building.elevators:
            previous_state = elevator.state
            elevator.update(dt)

            just_stopped = (previous_state == ElevatorState.MOVING and elevator.state == ElevatorState.STOPPED)
            if just_stopped:
                self._handle_arrival(elevator)

            just_opened = (previous_state == ElevatorState.DOOR_OPENING and elevator.state == ElevatorState.DOOR_OPEN)
            if just_opened:
                self._handle_doors_open(elevator)

            just_became_idle = ( previous_state == ElevatorState.DOOR_CLOSING and elevator.state == ElevatorState.IDLE)
            if just_became_idle:
                self._handle_doors_closed(elevator)

    def _handle_arrival(self, elevator) -> None:
        request = self._elevator_request.get(elevator.id)
        stage = self._elevator_stage.get(elevator.id)

        if request is not None:
            arrived_floor = (request.origin_floor if stage == "to_pickup" else request.destination_floor)
            self.events.emit(SimulationEvent(simulation_time=self.clock.simulation_time, event_type=EventType.ELEVATOR_ARRIVED,
                    data={"elevator_id": elevator.id, "floor": arrived_floor},
                    message=f"Elevator {elevator.id} arrived at F{arrived_floor + 1}"))

        elevator.open_doors()

    def _handle_doors_open(self, elevator) -> None:
        request = self._elevator_request.get(elevator.id)
        stage = self._elevator_stage.get(elevator.id)
        sim_time = self.clock.simulation_time

        if request is None:
            return

        if stage == "to_pickup":
            passenger = self.passengers.get(request.passenger_id)
            if passenger is not None and elevator.board(passenger):
                self.building.remove_waiting_passenger(passenger)
                passenger.board(sim_time)
                self.stats.record_pickup(passenger)
                request.mark_picked_up()

                self.events.emit(
                    SimulationEvent(simulation_time=sim_time, event_type=EventType.PASSENGER_BOARDED,
                        data={"elevator_id": elevator.id, "passenger_id": passenger.id},
                        message=f"Passenger {passenger.id} boarded elevator {elevator.id}"))

        elif stage == "to_dropoff":
            passenger = self.passengers.get(request.passenger_id)
            if passenger is not None:
                elevator.remove_passenger(passenger)
                passenger.arrive(sim_time)
                self.stats.record_completion(passenger)
                request.mark_completed()

                self.events.emit(
                    SimulationEvent(simulation_time=sim_time, event_type=EventType.PASSENGER_DROPPED_OFF,
                        data={"elevator_id": elevator.id, "passenger_id": passenger.id},
                        message=f"Passenger {passenger.id} arrived at F{passenger.destination_floor + 1}"))

    def _handle_doors_closed(self, elevator) -> None:
        request = self._elevator_request.get(elevator.id)
        stage = self._elevator_stage.get(elevator.id)

        if request is None:
            return

        if stage == "to_pickup":
            # Doors closed after pickup: head to the destination next.
            self._elevator_stage[elevator.id] = "to_dropoff"
            elevator.assign_target(request.destination_floor)
        elif stage == "to_dropoff":
            # Trip complete: free the elevator for a new assignment.
            del self._elevator_request[elevator.id]
            del self._elevator_stage[elevator.id]

    def _update_statistics(self) -> None:
        # Statistics themselves are updated incrementally and snapshot computes on demand.
        pass

    def get_statistics_snapshot(self) -> dict:
        waiting_count = len(self.building.all_waiting_passengers())
        inside_count = sum(len(e.passengers) for e in self.building.elevators)
        return self.stats.snapshot(waiting_count, inside_count, self.clock.simulation_time)

    def _get_elevator(self, elevator_id: int):
        for elevator in self.building.elevators:
            if elevator.id == elevator_id:
                return elevator
        raise KeyError(f"No elevator with id {elevator_id}")

    def reset(self) -> None:
        """Reset the entire simulation to a clean deterministic state."""
        self.clock.reset()
        self.clock.time_scale = self.config.default_time_scale
        self.building.reset()
        self.traffic_generator.reset()
        self.events.clear()
        self.stats.reset()
        self.passengers.clear()
        self.requests.clear()
        self._pending_requests.clear()
        self._elevator_request.clear()
        self._elevator_stage.clear()

