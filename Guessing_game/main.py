import os
from agents.entropic_agent import EntropicAgent
from agents.epsilon_agent import EpsilonInfoGainAgent
from agents.info_gain_agent import InfoGainAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from analytics.simulation import run_benchmark_simulation
from data.generator import generate_characters
from environments.system_opponent import SystemOpponent
from user_setting import play_with_user


def main():
    characters_df = generate_characters(n="max")
    model_path = "models/dqn_guess_who.zip"
    os.makedirs("models", exist_ok=True)

    print("             GUESS THE CHARACTER GAME             ")
    print("Choose mode:")
    print(" (1) Play single game vs System Opponent")
    print(" (2) Play interactive game vs Human User")
    print(" (3) Train RL Agent (DQN via Stable-Baselines3)")
    print(" (4) Benchmark All Agents (100 Epochs + Dashboard)")

    mode = input("\nEnter choice (1, 2, 3, or 4): ").strip()

    if mode == "1":
        opponent = SystemOpponent(characters_df)
        agent = EntropicAgent(
            search_space=characters_df, override_probability=0.1, max_steps=20
        )

        found = False
        while (
            not found
            and agent.step_counter < agent.max_steps
            and len(agent.search_space) > 0
        ):
            action = agent.take_turn()
            if action[0] == "error":
                print(f"Error: {action[1]}")
                break

            agent.step_counter += 1

            if action[0] == "guess":
                guess = action[1]
                print(
                    f"\nStep {agent.step_counter}: Agent guesses -> '{guess.name}'"
                )
                if guess.name == opponent.reveal().name:
                    print("Correct guess!")
                    found = True
                else:
                    print("Wrong guess!")
                    agent.rejected.add(guess.name)
                    agent.search_space = agent.search_space[
                        agent.search_space.index != guess.name
                    ]

            elif action[0] == "ask":
                f, v = action[1], action[2]
                agent.history.add((f, v))
                print(
                    f"\nStep {agent.step_counter}: Agent asks -> Is {f} = '{v}'?"
                )
                resp = opponent.answer(f, v)
                print(f"Opponent answers: {'Yes' if resp else 'No'}")
                agent.update_search_space(f, v, resp)
                print(f"Remaining candidates: {len(agent.search_space)}")

    elif mode == "2":
        play_with_user(EntropicAgent, characters_df)

    elif mode == "3":
        timesteps = input(
            "Enter training timesteps (default 100000): "
        ).strip()
        ts = int(timesteps) if timesteps.isdigit() else 100_000
        RLAgent.train_model(
            characters_df, total_timesteps=ts, save_path=model_path
        )

    elif mode == "4":
        epochs_input = input(
            "Enter number of simulation epochs (default 100): "
        ).strip()
        num_epochs = int(epochs_input) if epochs_input.isdigit() else 100

        # Build agent pool
        agent_pool = {
            "Random Agent": RandomAgent,
            "Cardinality Entropy": EntropicAgent,
            "Binary InfoGain": InfoGainAgent,
            "Epsilon InfoGain": EpsilonInfoGainAgent,
        }

        # Include RL agent if trained model exists
        if os.path.exists(model_path):
            agent_pool["DQN RL Agent"] = lambda space, max_steps: RLAgent(space, model_path=model_path, max_steps=max_steps)
        else:
            print(
                "\n[Note] Trained RL model not found. Run Mode (3) first to include Agent 5 in benchmarks!\n"
            )

        run_benchmark_simulation(
            agent_dict=agent_pool,
            characters_df=characters_df,
            num_epochs=num_epochs,
        )

    else:
        print("Invalid selection. Exiting.")

main()