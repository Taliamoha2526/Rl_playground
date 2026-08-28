import matplotlib.pyplot as plt
import numpy as np

def plot_comparative_dashboard(df_results):
    """ PLots dashboard of the results of each agent in the simualtion.
    Plot 1: Win rate comparison
    Plot 2: Average steps in a winning game
    Plot 3: Rolling win rate across games
    Plot 4: Search space contraction curve"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Multi-Agent Performance Comparison Dashboard", fontsize=15, fontweight="bold")

    agents = df_results["agent"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Win rates comparison plot
    win_rates = df_results.groupby("agent")["won"].mean() * 100
    axes[0, 0].bar(win_rates.index,  win_rates.values, color=colors[: len(agents)], edgecolor="black", alpha=0.85,)
    axes[0, 0].set_title("Overall Win Rate (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Win Rate %")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(win_rates.values):
        axes[0, 0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    # Average steps in a winning game
    wins_only = df_results[df_results["won"]]
    step_data = [wins_only[wins_only["agent"] == agent]["steps"] for agent in agents]
    axes[0, 1].boxplot(step_data, label=agents, patch_artist=True)
    axes[0, 1].set_title("Step Efficiency Distribution (Wins Only)", fontweight="bold")
    axes[0, 1].set_ylabel("Steps Taken")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Rolling win rate
    for idx, agent in enumerate(agents):
        agent_df = df_results[df_results["agent"] == agent].copy()
        agent_df["rolling_win"] = agent_df["won"].expanding().mean() * 100
        axes[1, 0].plot(agent_df["epoch"], agent_df["rolling_win"], label=agent, color=colors[idx % len(colors)], linewidth=2)

    axes[1, 0].set_title("Cumulative Win Rate Convergence", fontweight="bold")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Win Rate (%)")
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Search space contraction
    for idx, agent in enumerate(agents):
        agent_df = df_results[df_results["agent"] == agent]
        max_len = max(len(h) for h in agent_df["space_history"])
        padded_histories = np.array([np.pad(h, (0, max_len - len(h)), mode="edge") for h in agent_df["space_history"]])
        mean_trajectory = padded_histories.mean(axis=0)

        axes[1, 1].plot(range(len(mean_trajectory)), mean_trajectory, label=agent, color=colors[idx % len(colors)], linewidth=2)

    axes[1, 1].set_title("Search Space Contraction Curve", fontweight="bold")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Remaining Candidates")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()