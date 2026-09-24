from pathlib import Path
import os
import sys
# Path redirection fix
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in (CURRENT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents import *
from game_mechanics import *
from benchmark import *
from rl import *

MODEL_PATH = Path(__file__).resolve().parent / "rl_model.zip"

def rl_agent_ready():
    """Helper function when using the imported rl agent."""
    if not MODEL_PATH.exists():
        return False
    try:
        return True
    except ImportError:
        return False

def main():
    items = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    print("             Permutation Puzzle             ")
    print("Choose Mode:")
    print(" (1) Watch single game against system")
    print(" (2) Play against your chosen order")
    print(" (3) Train rl agent policy")
    print(" (4) Run benchmark simulations")

    mode = input("\n Enter your choice ").strip()

    if mode == "1":
        agents = { "1": ("RandomAgent", RandomAgent), "2": ("SystematicAgent", SystematicAgent),
                 "3": ("LogicAgent", LogicAgent), "4" : ("HybridAgent", HybridAgent), "5": ("KnuthAgent" , KnuthAgent)}
        #Include Rl when available
        if rl_agent_ready():
            agents["6"] = ("RLAgent", RLAgent)

        print("\n Select an agent to play against")
        for key, (name, _) in agents.items():
            print(f" ({key}) {name}")
        agent_choice = input("\n Enter your choice ").strip()
        if agent_choice in agents:
            name, object = agents[agent_choice]
            print(f"\n Starting game against: {name}")
            agent = object(items)
        else:
            raise ValueError("Please enter a valid agent")
        play(agent, items, max_steps=20)

    if mode == "2":
        agents = {"1": ("RandomAgent", RandomAgent), "2": ("SystematicAgent", SystematicAgent),
                  "3": ("LogicAgent", LogicAgent), "4": ("HybridAgent", HybridAgent), "5": ("KnuthAgent", KnuthAgent)}
        if rl_agent_ready():
            agents["6"] = ("RLAgent", RLAgent)

        print("\n Select an agent to play against")
        for key, (name, _) in agents.items():
            print(f" ({key}) {name}")
        agent_choice = input("\n Enter your choice ").strip()
        if agent_choice in agents:
            name, object = agents[agent_choice]
            print(f"\n Starting game against: {name}")
            agent = object(items)
        else:
            raise ValueError("Please enter valid agent!")
        play_with_human(agent, items, max_steps=20)

    if mode == "3":
        if rl_agent_ready():
            print(f"\n A trained model already exists at {MODEL_PATH} -- training will overwrite it.")
        steps_choice = input("\n Enter number of training timesteps (blank for default 1000000) ").strip()
        if steps_choice.isdigit():
            total_timesteps = int(steps_choice)
        else:
            total_timesteps = 1000000
        print(f"\n Training RL agent on {len(items)} elements for {total_timesteps} timesteps...")
        train_rl_agent(n_final=len(items), max_steps=20, total_timesteps=total_timesteps)
        print(f"\n Training complete. Model saved to {MODEL_PATH}")

    if mode == "4":
        epochs_choice = input("\n Enter number of simulation epochs ").strip()
        if epochs_choice.isdigit():
            epochs = int(epochs_choice)
        else:
            epochs = 100
        agents = [RandomAgent(items), SystematicAgent(items), LogicAgent(items), HybridAgent(items), KnuthAgent(items)]
        if rl_agent_ready():
            agents.append(RLAgent(items, model_path=MODEL_PATH))

        histories, results = simulate_agents(agents, items, epochs=epochs)
        plot_dashboard(results)

main()
