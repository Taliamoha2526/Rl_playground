import random
import numpy as np
import matplotlib.pyplot as plt

def simulate_agents(agents, items, secret_order=None, epochs=100, max_steps=20):
    """Compare the agent's performance on the same secrets for number of epochs.
    Returns:
      - histories: dict of each agent's name : list of episode histories (each guess and score pair)
      - results: dict of agent_name : list of steps to success
    """
    if secret_order is None:
        secret_order = random.sample(items, len(items))

    histories = {agent.__class__.__name__: [] for agent in agents}
    results = {agent.__class__.__name__: [] for agent in agents}

    for epoch in range(epochs):
        for agent in agents:
            agent.reset()
            episode_history = []
            steps = 0
            found = False

            while steps < max_steps and not found:
                steps += 1
                guess = agent.make_guess()
                score = sum([1 for i in range(len(secret_order)) if secret_order[i] == guess[i]])
                agent.receive_feedback(guess, score)
                episode_history.append((guess, score))

                if score == len(items):
                    found = True

            histories[agent.__class__.__name__].append(episode_history)
            results[agent.__class__.__name__].append(steps if found else max_steps)

    return histories, results

def plot_dashboard(results, max_steps=20):
    """
    Dashboard with:
      1. Survival curve (percentage solved by step)
      2. Consistency bar chart (average steps to success)
    """
    colors = {"RandomAgent": "dimgray", "SystematicAgent": "royalblue",
        "LogicAgent": "seagreen", "HybridAgent": "firebrick"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Percentage of wins by step
    for agent_name, steps_list in results.items():
        solved_counts = []
        total = len(steps_list)
        for step in range(1, max_steps+1):
            solved = sum(1 for s in steps_list if s <= step)
            solved_counts.append(100 * solved / total)
        axes[0].plot(range(1, max_steps+1), solved_counts,
                     label=agent_name, color=colors.get(agent_name,"black"))
    axes[0].set_title("Percentage of Episodes Solved by Step", fontsize=13)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Episodes Solved (%)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Plot 2: Average steps it took to win
    agent_names = list(results.keys())
    data = [results[name] for name in agent_names]
    bar_positions = np.arange(len(agent_names))
    bar_width = 0.6
    avg_steps = [np.mean(steps) for steps in data]

    axes[1].bar(bar_positions, avg_steps, width=bar_width,
                color=[colors.get(name, "black") for name in agent_names])
    axes[1].set_xticks(bar_positions)
    axes[1].set_xticklabels(agent_names, rotation=20)
    axes[1].set_title("Consistency of Agent Performance", fontsize=13)
    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Average Steps to Success")

    plt.tight_layout()
    plt.show()