"""Time keeping for the simulation.
The clock separates real (wall clock) time from simulation time so
that simulation speed can be scaled, paused, or run headless without
any dependency on a renderer's frame rate.
"""

from dataclasses import dataclass, field
@dataclass
class SimulationClock:
    """Tracks simulation time independent of rendering frame rate.
    The renderer is responsible for measuring real elapsed time and calling tick(real_dt),
    which returns the scaled simulation delta to feed into simulation.update(dt).
    Attributes:
        simulation_time: Total elapsed simulation time, in seconds.
        time_scale: Multiplier applied to real time to get sim time.
        paused: When True, tick always returns 0.0 simulation delta.
    """

    simulation_time: float = 0.0
    time_scale: float = 1.0
    paused: bool = False
    _last_delta: float = field(default=0.0, repr=False)

    def tick(self, real_dt: float) -> float:
        """Advance the clock given real elapsed seconds.
        Args:
            real_dt: Real seconds elapsed since last tick.
        Returns:
            The simulation time delta to apply this step (0.0 if paused).
        """
        if real_dt < 0:
            raise ValueError("real_dt must be non negative")

        if self.paused:
            self._last_delta = 0.0
            return 0.0

        simulation_delta = real_dt * self.time_scale
        self.simulation_time += simulation_delta
        self._last_delta = simulation_delta
        return simulation_delta

    def set_time_scale(self, scale: float) -> None:
        """Set the simulation speed multiplier."""
        if scale <= 0:
            raise ValueError("timescale must be positive")
        self.time_scale = scale

    def toggle_pause(self) -> None:
        """Flip the paused state."""
        self.paused = not self.paused

    def reset(self) -> None:
        """Reset simulation time to zero, keeping current timescale."""
        self.simulation_time = 0.0
        self._last_delta = 0.0
        self.paused = False
