"""Passenger traffic generation.
Responsible for creating passenger requests over simulation time.
Currently, implements only Random traffic generation, using a Poisson arrival process;
so passengers appear at simulation time correct intervals rather than on a fixed frame/tick schedule.
"""
import math
import random
from abc import ABC, abstractmethod
from typing import Optional
from elevator.settings_configs.config import SimulationConfig
from elevator.mechanics.passenger import Passenger


class PassengerGenerator(ABC):
    """General Interface for generating passengers over time."""

    @abstractmethod
    def update(self, simulation_time: float, dt: float) -> list[Passenger]:
        """Advance the generator and return any newly created passengers.
        Args:
            simulation_time: Current absolute simulation time.
            dt: Simulation seconds elapsed since the last update.
        Returns:
            A list (possibly empty) of newly created passengers.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal generator state."""
        raise NotImplementedError


class RandomTrafficGenerator(PassengerGenerator):
    """Generates passengers via a Poisson arrival process.
    Uses the config's passenger_arrival_rate as the expected number of arrivals per second.
    A seeded RNG makes sequences reproducible.
    """

    def __init__(self, config: SimulationConfig, seed: Optional[int] = None):
        self.config = config
        self._seed = seed if seed is not None else config.random_seed
        self._rng = random.Random(self._seed)
        self._next_passenger_id = 0
        self._time_since_last_check = 0.0

    def update(self, simulation_time: float, dt: float) -> list[Passenger]:
        """Generate zero or more passengers for this time slice.
        A Poisson process with rate lambda over a small interval dt,
        has a probability of one arrival of approximately lambda * dt.
        We sample against that probability each step.
        """
        new_passengers: list[Passenger] = []
        expected_arrivals = self.config.passenger_arrival_rate * dt
        count = self._sample_poisson(expected_arrivals)
        for _ in range(count):
            passenger = self._create_random_passenger(simulation_time)
            new_passengers.append(passenger)
        return new_passengers

    def _sample_poisson(self, lam: float) -> int:
        """Sample a Poisson distributed integer with mean lam."""
        if lam <= 0:
            return 0

        l_threshold = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self._rng.random()
            if p <= l_threshold:
                return k - 1

    def _create_random_passenger(self, simulation_time: float) -> Passenger:
        num_floors = self.config.num_floors
        origin = self._rng.randrange(num_floors)
        destination = self._rng.randrange(num_floors)
        while destination == origin:
            destination = self._rng.randrange(num_floors)

        passenger = Passenger(id=self._next_passenger_id, origin_floor=origin,
            destination_floor=destination, request_time=simulation_time)
        self._next_passenger_id += 1
        return passenger

    def reset(self) -> None:
        """Reset the RNG (to the original seed) and id counter."""
        self._rng = random.Random(self._seed)
        self._next_passenger_id = 0
        self._time_since_last_check = 0.0
