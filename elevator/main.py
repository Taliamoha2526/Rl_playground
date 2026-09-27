from settings_configs.config import SimulationConfig
from simulation.elevator_renderer import ElevatorRenderer
from simulation.simulation import ElevatorSimulation

def main() -> None:
    config = SimulationConfig(num_floors=10, num_elevators=3, elevator_capacity=6,
        max_elevator_speed=2.5, elevator_acceleration=1.0, elevator_deceleration=1.0,
        door_open_time=0.5, door_hold_time=2.0, door_close_time=0.5,
        passenger_arrival_rate=0.10, render_fps=60, default_time_scale=1.0)

    simulation = ElevatorSimulation(config=config)
    renderer = ElevatorRenderer(simulation=simulation)
    renderer.run()

main()