import random

def generate_secret_order(items):
    # The order to be guessed
    return random.sample(items, len(items))

def evaluate_guess(secret, guess):
    # Return score based on how many were right
    return sum([1 for i in range(len(secret)) if secret[i] == guess[i]])

def play(agent, items, max_steps=50):
    # Starts by generating completely new order
    secret = generate_secret_order(items)
    agent.reset()
    step_counter = 0
    found = False
    # Cue to keep playing
    while step_counter < max_steps and not found:
        step_counter += 1
        guess = agent.make_guess()
        score = evaluate_guess(secret, guess)
        agent.receive_feedback(guess, score)

        print(f"Step {step_counter}: Guess={guess}, Correct={score}")
        # Winning condition
        if score == len(items):
            print("Agent found the correct order!")
            found = True

def play_with_human(agent, items, max_steps=50):
    print("Human: secretly choose an order of the items.")
    print(f"Items: {items}")
    print("Do not reveal it. You will provide feedback each turn.")
    agent.reset()
    step_counter = 0
    found = False
    #Cue to keep playing
    while step_counter < max_steps and not found:
        step_counter += 1
        guess = agent.make_guess()
        print(f"Step {step_counter}: Agent guess = {guess}")

        # Human provides feedback manually
        score = int(input("Human: enter number of correct positions: "))
        agent.receive_feedback(guess, score)

        if score == len(items):
            print("Agent found the correct order!")
            found = True
