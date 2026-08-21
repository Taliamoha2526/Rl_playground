import random
from Guessing_game.agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, search_space, max_steps=20):
        super().__init__(search_space, max_steps)

    def take_turn(self):
        valid_candidates = self.search_space[
            ~self.search_space.index.isin(self.rejected)
        ]

        if len(valid_candidates) <= 1:
            return "guess", self.get_valid_random_guess()

        # Find unasked features and values
        unasked_pairs = []
        for col in self.search_space.columns:
            if col in self.asked_features:
                continue
            for val in self.search_space[col].dropna().unique():
                if (col, val) not in self.history:
                    unasked_pairs.append((col, val))

        if not unasked_pairs:
            return "guess", self.get_valid_random_guess()

        # Ask a completely random unasked question
        feature, value = random.choice(unasked_pairs)
        return "ask", feature, value