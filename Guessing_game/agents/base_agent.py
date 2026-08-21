import random
class BaseAgent:

    def __init__(self, search_space, max_steps=20):
        self.initial_space = search_space.copy()
        self.search_space = search_space.copy()
        self.max_steps = max_steps
        self.step_counter = 0
        self.asked_features = set()
        self.history = set()
        self.rejected = set()

    def get_valid_random_guess(self):
        valid_candidates = self.search_space[
            ~self.search_space.index.isin(self.rejected)
        ]
        if valid_candidates.empty:
            return None
        return valid_candidates.sample(1).iloc[0]

    def update_search_space(self, feature, value, response):
        if response:
            self.search_space = self.search_space[
                self.search_space[feature] == value
            ]
            self.asked_features.add(feature)
        else:
            self.search_space = self.search_space[
                (self.search_space[feature] != value)
                | (self.search_space[feature].isna())
            ]
            remaining_values = self.search_space[feature].unique()
            if len(remaining_values) <= 1:
                self.asked_features.add(feature)

    def take_turn(self):
        raise NotImplementedError