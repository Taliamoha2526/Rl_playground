from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PermutationPuzzleEnv
"""
Rl training loop + agent class to include in games

Two pieces:
  1. train_rl_agent() : trains a PPO policy on PermutationPuzzleEnv using stable baselines3
  2. RLAgent : a thin wrapper that uses the same method as the base agents, so a trained policy can be used directly

Note: RLAgent reconstructs the observations itself, independent of the environment, so it can be reused outside the context of training.
"""

def _make_env_fn(n, max_steps):
    def _fn():
        env = PermutationPuzzleEnv(n=n, max_steps=max_steps)
        return env
    return _fn

def train_rl_agent(n_final=10, max_steps=20, total_timesteps=500_000, n_envs=8,save_path="rl_model.zip"):
    """Train a PPO policy with the setup environment and save the model"""
    vec_env = make_vec_env(_make_env_fn(n_final, max_steps), n_envs=n_envs)
    model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=256, batch_size=256, gamma=0.99, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

class RLAgent:
    """Wraps a trained PPO policy in a regular agent to include in games and benchmarking."""
    def __init__(self, items, model_path="permutation_puzzle/rl_model.zip"):
        self.items = list(items)
        self.n = len(items)
        self._rng = np.random.default_rng()
        # Loading saved model from the path:
        file_name = Path(model_path).stem
        self.model = PPO.load(Path(__file__).parent / file_name)

        self._swap_pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self.reset()

    def reset(self):
        self.history = []
        self.best_guess = np.random.permutation(self.n)
        self.best_score = -1
        self.steps = 0
        self._first_guess_pending = True

    # Observations similar to the env
    def _compatibility_matrix(self):
        M = np.ones((self.n, self.n), dtype=np.float32)
        for past_guess, past_score in self.history:
            if past_score == 0:
                for i in range(self.n):
                    j = past_guess[i]
                    M[i, j] = 0.0
        return M

    def _best_guess_onehot(self):
        oh = np.zeros((self.n, self.n), dtype=np.float32)
        for pos, item in enumerate(self.best_guess):
            oh[pos, item] = 1.0
        return oh

    def _get_obs(self, max_steps=30):
        comp = self._compatibility_matrix().flatten()
        best = self._best_guess_onehot().flatten()
        best_score_norm = np.array([self.best_score / self.n], dtype=np.float32)
        steps_norm = np.array([self.steps / max_steps], dtype=np.float32)
        return np.concatenate([comp, best, best_score_norm, steps_norm]).astype(np.float32)

    # Agent playing similar to the heuristic agents
    def make_guess(self):
        self.steps += 1
        if self._first_guess_pending:
            self._first_guess_pending = False
            guess_idx = self.best_guess
        else:
            obs = self._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            i, j = self._swap_pairs[int(action)]
            guess_idx = self.best_guess.copy()
            guess_idx[i], guess_idx[j] = guess_idx[j], guess_idx[i]

        return [self.items[idx] for idx in guess_idx]

    def receive_feedback(self, guess, score):
        guess_idx = np.array([self.items.index(x) for x in guess])
        self.history.append((guess_idx, score))
        if score > self.best_score:
            self.best_score = score
            self.best_guess = guess_idx

