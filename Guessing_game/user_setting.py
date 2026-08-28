def get_user_boolean(prompt: str) -> bool:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        print("Invalid input! Answer with 'y' or 'n'.")


def play_with_user(agent_cls, characters_df, max_steps=20):
    print("      GUESS THE CHARACTER GAME            ")

    agent = agent_cls(characters_df, max_steps=max_steps)

    print("\nChoose one character secretly. The agent will try to guess it!")
    input("Press ENTER when you are ready to start...")

    found = False
    #Cue to keep playing
    while not found and agent.step_counter < agent.max_steps and len(agent.search_space) > 0:
        action = agent.take_turn()

        if action[0] == "error":
            print(f"\nGame Over: {action[1]}")
            break

        agent.step_counter += 1

        if action[0] == "guess":
            guess_series = action[1]
            guess_idx = guess_series.name

            print( f"\n--- Step {agent.step_counter} ---"
                f"\nAgent guesses: ** {guess_idx} **")

            is_correct = get_user_boolean("Is this your character? (y/n): ")

            if is_correct:
                print(f"\nCorrect! Agent guessed '{guess_idx}' in {agent.step_counter} steps!")
                found = True
            else:
                print("Incorrect guess. Updating candidate space...")
                agent.rejected.add(guess_idx)
                agent.search_space = agent.search_space[agent.search_space.index != guess_idx]
                print(f"Remaining candidates: {len(agent.search_space)}")

        elif action[0] == "ask":
            feature, value = action[1], action[2]
            agent.history.add((feature, value))

            print(f"\n--- Step {agent.step_counter} ---"
                f"\nAgent asks: Is {feature} = '{value}'?")
            #Get the user's answer and update based on it.
            response = get_user_boolean("(y/n): ")
            agent.update_search_space(feature, value, response)
            print(f"Remaining candidates: {len(agent.search_space)}")

    if not found:
        if len(agent.search_space) == 0:
            print("Game ended: No remaining characters match your answers (Contradiction).")
        elif agent.step_counter >= agent.max_steps:
            print(f"Game ended: Reached maximum limit of {agent.max_steps} steps.")