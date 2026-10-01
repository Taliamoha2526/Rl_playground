import dataclasses
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # adjust if this file moves

from settings_configs.config import SimulationConfig
from simulation.simulation import ElevatorSimulation, CapacityAwareSimulation

def print_welcome() -> None:
    print("     ELEVATOR SIMULATOR     ")
    print("A time based elevator simulation. You'll pick a building,\n"
        "configuration, a dispatch strategy, and whether to watch it\n"
        "live or run it headless and save the results.\n")

CONFIG_FIELDS = [("num_floors", "Number of floors", int),
    ("num_elevators", "Number of elevators", int),
    ("elevator_capacity", "Elevator capacity", int),
    ("passenger_arrival_rate", "Passenger arrival rate (passengers/sec)", float),
    ("random_seed", "Random seed (for reproducible traffic)", int)]


def prompt_config() -> SimulationConfig:
    defaults = SimulationConfig()
    overrides = {}

    print("Enter configuration values, or press Enter to use the default shown.")
    print("An invalid value falls back to the configuration's default.\n")

    for field_name, label, cast in CONFIG_FIELDS:
        default_value = getattr(defaults, field_name)
        raw = input(f"  {label} ({default_value}): ").strip()
        if not raw:
            continue
        try:
            overrides[field_name] = cast(raw)
        except ValueError:
            print(f"Couldn't parse '{raw}' as {cast.__name__}; using default ({default_value}).")

    try:
        config = SimulationConfig(**overrides)
    except ValueError as e:
        # If the value failed the config's own value restrictions
        print(f"\nConfiguration rejected ({e}); falling back to full defaults.\n")
        config = SimulationConfig()

    return config

SIMULATION_CHOICES = {"1": ("Nearest available dispatcher (baseline)", lambda config: ElevatorSimulation(config=config)),
    "2": ("Capacity aware dispatcher (multi stop)", lambda config: CapacityAwareSimulation(config=config))}


def prompt_simulation_choice(config: SimulationConfig):
    print("\nChoose a dispatch strategy:")
    for key, (label, _factory) in SIMULATION_CHOICES.items():
        print(f"  {key}) {label}")

    choice = input("Enter a number: ").strip() or "1"
    if choice not in SIMULATION_CHOICES:
        print(f"  Unrecognized choice '{choice}'; using option 1.")
        choice = "1"

    label, factory = SIMULATION_CHOICES[choice]
    print(f" Running : {label}")
    return factory(config)

def prompt_mode() -> str:
    print("\nHow do you want to run it?")
    print("  1) Watch it live (pygame window)")
    print("  2) Run headless and save the results")
    choice = input("Enter a number: ").strip() or "1"
    return "headless" if choice == "2" else "render"


def run_with_renderer(simulation) -> None:
    from simulation.elevator_renderer import ElevatorRenderer
    renderer = ElevatorRenderer(simulation)
    renderer.run()


def run_headless(simulation, config: SimulationConfig, output_dir: str = "results") -> str:
    raw_steps = input("\nHow many simulation steps to run? (2000): ").strip()
    try:
        num_steps = int(raw_steps) if raw_steps else 2000
    except ValueError:
        print(f"  Couldn't parse '{raw_steps}' as an integer; using 2000.")
        num_steps = 2000

    dt = 0.1
    snapshot_every = 50

    print(f"Running {num_steps} steps at dt={dt}s "
          f"({num_steps * dt:.1f} simulated seconds)...")

    snapshots = []
    for step in range(1, num_steps + 1):
        simulation.update(dt)
        if step % snapshot_every == 0:
            snapshots.append(simulation.get_statistics_snapshot())

    final_stats = simulation.get_statistics_snapshot()
    print("# Warning including the full events log might produce an incredibly large file.")
    include_events = input("Include the full event log in the output? [y/n]:").strip().lower() == "y"
    events = None
    if include_events:
        events = [{"time": e.simulation_time, "type": e.event_type.name, "message": e.message}
            for e in simulation.events.history]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"run_{timestamp}.json")

    payload = {"config": dataclasses.asdict(config),
        "num_steps": num_steps,
        "dt": dt,
        "final_stats": final_stats,
        "stats_timeseries": snapshots}
    if events is not None:
        payload["events"] = events

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nDone. Completed {final_stats['completed_passengers']}/"
          f"{final_stats['total_passengers_generated']} passengers, "
          f"avg wait {final_stats['average_waiting_time']:.1f}s.")
    print(f"Saved to {output_path}")
    return output_path

def main() -> None:
    print_welcome()
    config = prompt_config()
    simulation = prompt_simulation_choice(config)
    mode = prompt_mode()

    if mode == "render":
        run_with_renderer(simulation)
    else:
        run_headless(simulation, config)

main()