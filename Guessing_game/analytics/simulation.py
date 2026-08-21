import random
import pandas as pd
from Guessing_game.analytics.dashboard import plot_comparative_dashboard
from Guessing_game.environments.system_opponent import SystemOpponent


def run_benchmark_simulation(
    agent_dict, characters_df, num_epochs=100, max_steps=20
):
    all_logs = []

    print(
        f"\nStarting Comparative Benchmark: {len(agent_dict)} Agents over {num_epochs} Epochs...\n"
    )

    for epoch in range(1, num_epochs + 1):
        random.seed(epoch)
        secret_name = random.choice(characters_df.index)

        for agent_name, agent_cls in agent_dict.items():
            opponent = SystemOpponent(
                characters_df, secret_character_name=secret_name
            )
            agent = agent_cls(characters_df, max_steps=max_steps)

            epoch_data = {
                "epoch": epoch,
                "agent": agent_name,
                "target": secret_name,
                "won": False,
                "steps": 0,
                "guesses_made": 0,
                "questions_asked": 0,
                "space_history": [len(characters_df)],
                "outcome": "MAX_STEPS_REACHED",
            }

            found = False

            while (
                not found
                and agent.step_counter < agent.max_steps
                and len(agent.search_space) > 0
            ):
                action = agent.take_turn()

                if action[0] == "error":
                    epoch_data["outcome"] = "CONTRADICTION_ERROR"
                    break

                agent.step_counter += 1

                if action[0] == "guess":
                    epoch_data["guesses_made"] += 1
                    guess = action[1]
                    guess_idx = guess.name

                    if guess_idx == opponent.reveal().name:
                        found = True
                        epoch_data["won"] = True
                        epoch_data["outcome"] = "SUCCESS"
                    else:
                        agent.rejected.add(guess_idx)
                        agent.search_space = agent.search_space[
                            agent.search_space.index != guess_idx
                        ]

                elif action[0] == "ask":
                    epoch_data["questions_asked"] += 1
                    feature, value = action[1], action[2]
                    agent.history.add((feature, value))

                    response = opponent.answer(feature, value)
                    agent.update_search_space(feature, value, response)

                epoch_data["space_history"].append(len(agent.search_space))

            epoch_data["steps"] = agent.step_counter
            all_logs.append(epoch_data)

    df_results = pd.DataFrame(all_logs)

    summary = (
        df_results.groupby("agent")
        .agg(
            Win_Rate=(
                "won",
                lambda x: f"{(x.sum() / len(x)) * 100:.1f}%",
            ),
            Avg_Steps=("steps", "mean"),
            Questions_Asked=("questions_asked", "mean"),
            Guesses_Made=("guesses_made", "mean"),
        )
        .reset_index()
    )

    print("                      BENCHMARK SUMMARY                      ")
    print(summary.to_string(index=False))

    plot_comparative_dashboard(df_results)
    return df_results