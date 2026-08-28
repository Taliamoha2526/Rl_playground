import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GuessWhoEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, characters_df, max_steps=100):
        super().__init__()
        self.df = characters_df.copy()
        self.max_steps = max_steps

        # Index binary questions
        self.questions = []
        for col in self.df.columns:
            unique_vals = self.df[col].dropna().unique()
            for val in sorted(unique_vals):
                self.questions.append((col, val))

        self.num_questions = len(self.questions)
        self.num_candidates = len(self.df)

        self.action_space = spaces.Discrete(self.num_questions + self.num_candidates)

        # Observation Space: Asked history, remaining candidate ratio, and dynamic split ratios per question
        self.observation_space = spaces.Dict({"asked_mask": spaces.MultiBinary(self.num_questions),
            "remaining_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "split_ratios": spaces.Box(low=0.0, high=1.0, shape=(self.num_questions,), dtype=np.float32)})

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #Ensure reset to new episode/game.
        self.secret_idx = random.randint(0, self.num_candidates - 1)
        self.secret_character = self.df.iloc[self.secret_idx]

        self.candidate_mask = np.ones(self.num_candidates, dtype=np.int8)
        self.asked_mask = np.zeros(self.num_questions, dtype=np.int8)
        self.step_counter = 0

        return self._get_obs(), {}

    def _get_obs(self):
        rem_count = self.candidate_mask.sum()
        rem_ratio = np.array([rem_count / self.num_candidates], dtype=np.float32)

        # Dynamic Split Ratio
        split_ratios = np.zeros(self.num_questions, dtype=np.float32)
        if rem_count > 0:
            active_df = self.df.iloc[np.where(self.candidate_mask == 1)[0]]
            for i, (feat, val) in enumerate(self.questions):
                if self.asked_mask[i] == 1:
                    split_ratios[i] = 0.0
                else:
                    matches = (active_df[feat] == val).sum()
                    split_ratios[i] = matches / rem_count

        return {"asked_mask": self.asked_mask.copy(),
            "remaining_ratio": rem_ratio, "split_ratios": split_ratios}

    def step(self, action):
        self.step_counter += 1
        terminated = False
        truncated = self.step_counter >= self.max_steps
        reward = -0.05  # Step penalty

        rem_count = self.candidate_mask.sum()

        # Enforce penalty for premature candidate guessing when candidates > 1
        if action >= self.num_questions and rem_count > 1:
            return self._get_obs(), -5.0, False, truncated, {}

        rem_before = max(1, rem_count)

        if action < self.num_questions:
            feature, value = self.questions[action]

            if self.asked_mask[action] == 1:
                reward = -2.0  # Penalty for re-asking
            else:
                self.asked_mask[action] = 1
                is_match = self.secret_character[feature] == value

                if is_match:
                    match_cond = (self.df[feature] == value).values
                else:
                    match_cond = ((self.df[feature] != value) | (self.df[feature].isna())).values

                self.candidate_mask = self.candidate_mask & match_cond.astype(int)
                rem_after = max(1, self.candidate_mask.sum())

                # Logarithmic entropy reward based on the split contraction
                if rem_before > rem_after:
                    bits_gained = math.log2(rem_before / rem_after)
                    reward += bits_gained

        # Guessing
        else:
            guess_idx = action - self.num_questions
            if guess_idx == self.secret_idx:
                reward = 10.0
                terminated = True
            else:
                reward = -5.0
                self.candidate_mask[guess_idx] = 0

        # Auto terminate if search space is narrowed to target but not yet terminated.
        if self.candidate_mask.sum() == 1 and not terminated:
            rem_idx = np.where(self.candidate_mask == 1)[0][0]
            if rem_idx == self.secret_idx:
                reward = 10.0
                terminated = True

        return self._get_obs(), reward, terminated, truncated, {}