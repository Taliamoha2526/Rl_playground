import math
from Guessing_game.agents.base_agent import BaseAgent


class InfoGainAgent(BaseAgent):
    """Selects question based on maximum binary information gain (closest to 50/50 split)."""

    def __init__(self, search_space, max_steps=20):
        super().__init__(search_space, max_steps)

    def take_turn(self):
        valid_candidates = self.search_space[
            ~self.search_space.index.isin(self.rejected)
        ]
        total_candidates = len(valid_candidates)

        if total_candidates == 0:
            return "error", "No valid candidates remain."

        # End-game or remaining = 1
        turns_left = self.max_steps - self.step_counter
        if total_candidates <= turns_left or total_candidates <= 2:
            return "guess", valid_candidates.sample(1).iloc[0]

        best_score = -1.0
        best_pair = (None, None)

        for feature in self.search_space.columns:
            if feature in self.asked_features:
                continue

            unique_vals = self.search_space[feature].dropna().unique()
            for val in unique_vals:
                if (feature, val) in self.history:
                    continue

                match_count = (valid_candidates[feature] == val).sum()
                p = match_count / total_candidates

                if 0 < p < 1:
                    # Shannon binary entropy
                    binary_entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
                    if binary_entropy > best_score:
                        best_score = binary_entropy
                        best_pair = (feature, val)

        if best_pair[0] is not None:
            return "ask", best_pair[0], best_pair[1]

        return "guess", self.get_valid_random_guess()