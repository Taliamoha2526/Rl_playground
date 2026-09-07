from agents import *
from game_mechanics import *
from benchmark import *

def main():
    items = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    print("             Permutation Puzzle             ")
    print("Choose Mode:")
    print(" (1) Watch single game against system")
    print(" (2) Play against your chosen order")
    print(" (3) Run benchmark simulations")

    mode = input("\n Enter your choice ").strip()

    if mode == "1":
        agents = { "1": ("RandomAgent", RandomAgent), "2": ("SystematicAgent", SystematicAgent),
                 "3": ("LogicAgent", LogicAgent), "4" : ("HybridAgent", HybridAgent)}
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
                  "3": ("LogicAgent", LogicAgent), "4": ("HybridAgent", HybridAgent)}
        print("\n Select an agent to play against")
        for key, (name, _) in agents.items():
            print(f" ({key}) {name}")
        agent_choice = input("\n Enter your choice ").strip()
        if agent_choice in agents:
            name, object = agents[agent_choice]
            print(f"\n Starting game against: {name}")
            agent = object(items)
        else:
            raise ValueError("Please enter vaild agent!")
        play_with_human(agent, items, max_steps=20)
    if mode == "3":
        epochs_choice = input("\n Enter number of simulation epochs ").strip()
        if epochs_choice.isdigit():
            epochs = int(epochs_choice)
        else:
            epochs = 100
        agents = [RandomAgent(items), SystematicAgent(items), LogicAgent(items), HybridAgent(items)]
        histories, results = simulate_agents(agents, items, epochs=epochs)
        plot_dashboard(results)


main()

