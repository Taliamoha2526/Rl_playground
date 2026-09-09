import numpy as np
import gymnasium as gym
from gymnasium import spaces

"""
- Action space = Discrete(n*(n-1)/2): choose which pair of items to swap relative to the current best known guess. 
Similar to the logic of the heuristics agents, but more computationally efficient.
- Observation :
    - n x n compatibility matrix (flattened): is item j allowed at position i given everything observed so far 
    - current best guess, one hot encoded for faster and simpler comparison
    - best score so far / n
    - steps taken / max_steps
- Reward: small step penalty + big terminal bonus on solving. 
To emphasize the greater goal is to get the final solution faster and not to be mislead by score fluctuations
- A fresh secret order is drawn every episode in reset()
"""
class PermutationPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, n=10, max_steps=20, step_penalty=-0.02, solve_bonus=1.0):
        super().__init__()
        self.n = n
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.solve_bonus = solve_bonus

        # Precompute the list of (i, j) swap pairs that action indices map to.
        self._swap_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.action_space = spaces.Discrete(len(self._swap_pairs))

        # Observation: compatibility matrix (n*n) + encoded best guess (n*n) + normalized best score (1) + normalized steps (1)
        obs_len = self.n * self.n + self.n * self.n + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)
        self._rng = np.random.default_rng()

    # Reusing core game mechanics
    def _score(self, guess, secret):
        return int(np.sum(guess == secret))

    def _compatibility_matrix(self):
        """M[i, j] = 1.0 if item j at position i is still consistent with every historical (guess, score) pair.
        M[i, j] = 0.0 if the item is restricted from the position"""
        M = np.ones((self.n, self.n), dtype=np.float32)
        # For each past guess, position i is being tested for element j:
        for past_guess, past_score in self.history:
            if past_score == 0:
                # If the score is zero, immediately restrict all elements from their current positions
                # (as it is impossible for them to be correct)
                for i in range(self.n):
                    j = past_guess[i]
                    M[i, j] = 0.0
        return M

    def _best_guess_onehot(self):
        oh = np.zeros((self.n, self.n), dtype=np.float32)
        for pos, item in enumerate(self.best_guess):
            oh[pos, item] = 1.0
        return oh

    def _get_obs(self):
        comp = self._compatibility_matrix().flatten()
        best = self._best_guess_onehot().flatten()
        best_score_norm = np.array([self.best_score / self.n], dtype=np.float32)
        steps_norm = np.array([self.steps / self.max_steps], dtype=np.float32)
        return np.concatenate([comp, best, best_score_norm, steps_norm]).astype(np.float32)

    def _get_info(self):
        return {"steps": self.steps, "best_score": self.best_score, "best_guess": self.best_guess.copy()}

    # Gymnasium settings
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset secret for each episode
        self.secret = self._rng.permutation(self.n)
        self.history = []
        self.steps = 0
        self.best_guess = self._rng.permutation(self.n)
        self.best_score = self._score(self.best_guess, self.secret)
        self.history.append((self.best_guess.copy(), self.best_score))

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1
        i, j = self._swap_pairs[int(action)]
        # Perform the selected swap action
        candidate = self.best_guess.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]

        score = self._score(candidate, self.secret)
        self.history.append((candidate.copy(), score))

        if score > self.best_score:
            self.best_score = score
            self.best_guess = candidate

        solved = score == self.n
        terminated = bool(solved)
        truncated = bool(self.steps >= self.max_steps and not solved)

        reward = self.step_penalty
        if solved:
            # Reward faster solutions
            reward = self.solve_bonus * (1.0 - 0.5 * self.steps / self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"Step {self.steps}: best_guess={self.best_guess}, best_score={self.best_score}")
