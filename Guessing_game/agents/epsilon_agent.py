import random
from Guessing_game.agents.info_gain_agent import InfoGainAgent

class EpsilonInfoGainAgent(InfoGainAgent):

    def __init__(self, search_space, epsilon=0.15, max_steps=20):
        super().__init__(search_space, max_steps)
        self.epsilon = epsilon

    def take_turn(self):
        # Added random exploration based on epsilon probability
        if random.random() < self.epsilon:
            unasked_pairs = []
            for col in self.search_space.columns:
                if col in self.asked_features:
                    continue
                for val in self.search_space[col].dropna().unique():
                    if (col, val) not in self.history:
                        unasked_pairs.append((col, val))

            if unasked_pairs:
                f, v = random.choice(unasked_pairs)
                return "ask", f, v

        # Usual greedy play of Infogain
        return super().take_turn()