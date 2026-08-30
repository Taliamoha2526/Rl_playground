import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GuessWhoEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, characters_df, max_steps=100):
        super().__init__()
        self.df = characters_df.copy()
        self.max_steps = max_steps

        # Index binary questions.
        self.questions = []
        self.feature_bounds = []
        for col in self.df.columns:
            start = len(self.questions)
            unique_vals = self.df[col].dropna().unique()
            for val in sorted(unique_vals):
                self.questions.append((col, val))
            self.feature_bounds.append((start, len(self.questions)))

        self.num_questions = len(self.questions)
        self.num_candidates = len(self.df)

        # Use the question matrix mask
        self.question_matrix = np.zeros((self.num_candidates, self.num_questions), dtype=np.float32)
        self.nan_masks = np.zeros((self.num_candidates, self.num_questions), dtype=bool)
        for j, (feat, val) in enumerate(self.questions):
            self.question_matrix[:, j] = (self.df[feat] == val).values
            self.nan_masks[:, j] = self.df[feat].isna().values

        # The action space focuses on questions
        self.action_space = spaces.Discrete(self.num_questions)

        # Observation Space: Asked history, remaining candidate ratio, and dynamic split ratios per question
        self.observation_space = spaces.Dict({"asked_mask": spaces.MultiBinary(self.num_questions),
            "remaining_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "split_ratios": spaces.Box(low=0.0, high=1.0, shape=(self.num_questions,), dtype=np.float32)})

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #Ensure reset to new episode/game.
        self.secret_idx = int(self.np_random.integers(0, self.num_candidates))
        self.secret_character = self.df.iloc[self.secret_idx]

        self.candidate_mask = np.ones(self.num_candidates, dtype=np.int8)
        self.asked_mask = np.zeros(self.num_questions, dtype=np.int8)
        self.step_counter = 0
        self.sync_resolved_features()

        return self._get_obs(), {}

    def sync_resolved_features(self):
        """Mark every (feature, value) action for a column as 'asked' once they are resolved.
        Can be through direct confirmation, binary deduction, values exhaustion.
        """
        rem_count = self.candidate_mask.sum()
        if rem_count <= 0:
            return
        matches = self.candidate_mask.astype(np.float32) @ self.question_matrix
        for start, end in self.feature_bounds:
            seg = matches[start:end]
            if seg.size == 0:
                continue
            # Locked if one value now covers every active candidate
            if seg.max() >= rem_count - 1e-6 or seg.sum() <= 1e-6:
                self.asked_mask[start:end] = 1

    def _get_obs(self):
        rem_count = self.candidate_mask.sum()
        rem_ratio = np.array([rem_count / self.num_candidates], dtype=np.float32)

        # Dynamic Split Ratio using the question m
        split_ratios = np.zeros(self.num_questions, dtype=np.float32)
        if rem_count > 0:
            matches = self.candidate_mask.astype(np.float32) @ self.question_matrix
            split_ratios = (matches / rem_count).astype(np.float32)
            split_ratios[self.asked_mask == 1] = 0.0

        return {"asked_mask": self.asked_mask.copy(),
            "remaining_ratio": rem_ratio, "split_ratios": split_ratios}

    def step(self, action):
        self.step_counter += 1
        terminated = False
        truncated = self.step_counter >= self.max_steps
        reward = -0.05  # Step penalty

        rem_before = max(1, self.candidate_mask.sum())
        feature, value = self.questions[action]

        if self.asked_mask[action] == 1:
            reward = -2.0  # Penalty for re-asking / asking an already-resolved feature
        else:
            self.asked_mask[action] = 1
            is_match = self.secret_character[feature] == value

            match_bool = self.question_matrix[:, action].astype(bool)
            if is_match:
                match_cond = match_bool
            else:
                match_cond = (~match_bool) | self.nan_masks[:, action]

            self.candidate_mask = self.candidate_mask & match_cond.astype(int)
            rem_after = max(1, self.candidate_mask.sum())

            # Logarithmic entropy reward based on the split contraction
            if rem_before > rem_after:
                bits_gained = math.log2(rem_before / rem_after)
                reward += bits_gained

        # Terminal: pool narrowed to a single candidate or every feature value for the secret character is known.
        if self.candidate_mask.sum() == 1 and not terminated:
            rem_idx = np.where(self.candidate_mask == 1)[0][0]
            if rem_idx == self.secret_idx:
                reward = 10.0
                terminated = True

        # Sync the learnt features
        self.sync_resolved_features()

        return self._get_obs(), reward, terminated, truncated, {}