import math
import random
from Guessing_game.agents.base_agent import BaseAgent


class EntropicAgent(BaseAgent):
    def __init__(self, search_space, override_probability=0.1, max_steps=20):
        super().__init__(search_space, max_steps)
        self.override_probability = override_probability

    def compute_entropy(self, distribution):
        #Compute entropy for each feature and it's possible values.
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        return -sum((count / total) * math.log2(count / total) for count in distribution.values() if count > 0 )

    def get_distribution(self, feature):
        return dict(self.search_space[feature].value_counts())

    def choose_feature_entropy(self):
        feature_scores = {}
        last_asked_feature = (list(self.history)[-1][0] if len(self.history) > 0 else None)

        for feature in self.search_space.columns:
            if feature in self.asked_features:
                continue
            #Get the distribution of values for the features.
            distribution = self.get_distribution(feature)
            if len(distribution) <= 1:
                self.asked_features.add(feature)
                continue

            entropy = self.compute_entropy(distribution)

            # Prevent feature lock loops
            if feature == last_asked_feature:
                entropy *= 0.1

            feature_scores[feature] = entropy

        if not feature_scores:
            return None

        # Choose the one with the highest entropy.
        max_score = max(feature_scores.values())
        #Choose randomly from the top if there is a tie
        top_features = [f for f, score in feature_scores.items()if math.isclose(score, max_score)]
        return random.choice(top_features)

    def choose_feature_value_frequency(self, feature):
        distribution = self.get_distribution(feature)
        #Use the value distribution to choose which specific value to ask about.
        for value in sorted(distribution, key=distribution.get, reverse=True):
            if (feature, value) not in self.history:
                return value
        return None

    def take_turn(self):
        valid_candidates = self.search_space[~self.search_space.index.isin(self.rejected)]
        valid_candidates_count = len(valid_candidates)

        if valid_candidates_count == 0:
            return "error", "No valid candidates remain."

        # End-game trigger
        turns_left = self.max_steps - self.step_counter
        if valid_candidates_count <= turns_left or valid_candidates_count <= 2:
            return "guess", valid_candidates.sample(1).iloc[0]

        # Random feature and value based on override probability
        if random.random() < self.override_probability:
            unasked_features = [col for col in self.search_space.columns if col not in self.asked_features]
            if unasked_features:
                r_feat = random.choice(unasked_features)
                r_val = random.choice(list(self.search_space[r_feat].dropna().unique()))
                if (r_feat, r_val) not in self.history:
                    return "ask", r_feat, r_val

        feature = self.choose_feature_entropy()
        if feature is None:
            return "guess", self.get_valid_random_guess()

        value = self.choose_feature_value_frequency(feature)
        if value is None:
            self.asked_features.add(feature)
            return "guess", self.get_valid_random_guess()

        return "ask", feature, value