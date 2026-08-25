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
        agent_options = {
            "1": ("Random Agent", RandomAgent),
            "2": ("Cardinality Entropy Agent", EntropicAgent),
            "3": ("Binary InfoGain Agent", InfoGainAgent),
            "4": ("Epsilon InfoGain Agent", EpsilonInfoGainAgent),
        }

        if os.path.exists(model_path):
            agent_options["5"] = "DQN RL Agent", lambda space, max_steps=20: RLAgent(space, model_path=model_path, max_steps=max_steps)

        print("\nSelect an Agent to watch play:")
        for key, (name, _) in agent_options.items():
            print(f" ({key}) {name}")

        agent_choice = input("\nEnter choice: ").strip()

        if agent_choice in agent_options:
            selected_name, agent_factory = agent_options[agent_choice]
            print(f"\nStarting game watching: {selected_name}...")
            agent = agent_factory(characters_df, max_steps=20)
        else:
            print("Invalid choice. Defaulting to Cardinality Entropy Agent.")
            agent = EntropicAgent(search_space=characters_df, max_steps=20)

        opponent = SystemOpponent(characters_df)
        found = False

        while (not found and agent.step_counter < agent.max_steps and len(agent.search_space) > 0):
            action = agent.take_turn()
            if action[0] == "error":
                print(f"Error: {action[1]}")
                break

            agent.step_counter += 1

            if action[0] == "guess":
                guess = action[1]
                guess_name = guess.name if hasattr(guess, "name") else str(guess)
                print(f"\nStep {agent.step_counter}: Agent guesses -> '{guess_name}'")
                if guess_name == opponent.reveal().name:
                    print("Correct guess!")
                    found = True
                else:
                    print("Wrong guess!")
                    agent.rejected.add(guess_name)
                    agent.search_space = agent.search_space[agent.search_space.index != guess_name]

            elif action[0] == "ask":
                f, v = action[1], action[2]
                agent.history.add((f, v))
                print(f"\nStep {agent.step_counter}: Agent asks -> Is {f} = '{v}'?")
                resp = opponent.answer(f, v)
                print(f"Opponent answers: {'Yes' if resp else 'No'}")
                agent.update_search_space(f, v, resp)
                print(f"Remaining candidates: {len(agent.search_space)}")

        if not found:
            print(f"\nAgent ran out of turns or candidates! Secret character was '{opponent.reveal().name}'.")

    elif mode == "2":
        agent_options = {"1": ("Random Agent", RandomAgent),
            "2": ("Cardinality Entropy Agent", EntropicAgent),
            "3": ("Binary InfoGain Agent", InfoGainAgent),
            "4": ("Epsilon InfoGain Agent", EpsilonInfoGainAgent)}

        if os.path.exists(model_path):
            agent_options["5"] = ("DQN RL Agent",lambda space, max_steps=20: RLAgent(space, model_path=model_path, max_steps=max_steps))

        print("\nSelect an Agent to play against:")
        for key, (name, _) in agent_options.items():
            print(f" ({key}) {name}")

        agent_choice = input("\nEnter choice: ").strip()

        if agent_choice in agent_options:
            selected_name, selected_agent_cls = agent_options[agent_choice]
            print(f"\nStarting game against: {selected_name}...")
            play_with_user(selected_agent_cls, characters_df)
        else:
            print("Invalid agent choice. Defaulting to Cardinality Entropy Agent.")
            play_with_user(EntropicAgent, characters_df)

    elif mode == "3":
        timesteps = input("Enter training timesteps (default 100000): ").strip()
        ts = int(timesteps) if timesteps.isdigit() else 100_000
        RLAgent.train_model(characters_df, total_timesteps=ts, save_path=model_path)

    elif mode == "4":
        epochs_input = input("Enter number of simulation epochs (default 100): ").strip()
        num_epochs = int(epochs_input) if epochs_input.isdigit() else 100

        # Build agent pool
        agent_pool = {"Random Agent": RandomAgent,
            "Cardinality Entropy": EntropicAgent,
            "Binary InfoGain": InfoGainAgent,
            "Epsilon InfoGain": EpsilonInfoGainAgent}

        # Include RL agent if trained model exists
        if os.path.exists(model_path):
            agent_pool["DQN RL Agent"] = lambda space, max_steps: RLAgent(space, model_path=model_path, max_steps=max_steps)
        else:
            print("\n[Note] Trained RL model not found. Run Mode (3) first to include Agent 5 in benchmarks!\n")

        run_benchmark_simulation(agent_dict=agent_pool, characters_df=characters_df, num_epochs=num_epochs,)

    else:
        print("Invalid selection. Exiting.")

main()