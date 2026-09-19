from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from env import PermutationPuzzleEnv
"""
Rl training loop + agent class to include in games

Two pieces:
  1. train_rl_agent() : trains a DQN agent on PermutationPuzzleEnv using stable baselines3
  2. RLAgent : a thin wrapper that uses the same method as the base agents, so a trained policy can be used directly

Note: RLAgent reconstructs the observations itself, independent of the environment, so it can be reused outside the context of training.
"""

def _make_env_fn(n, max_steps, delta_reward_scale=0.1):
    def _fn():
        env = PermutationPuzzleEnv(n=n, max_steps=max_steps, delta_reward_scale=delta_reward_scale)
        return env
    return _fn

def train_rl_agent(n_final=10, max_steps=20, total_timesteps=500_000, n_envs=8,save_path="rl_model.zip"):
    """Train a DQN policy with the setup environment and save the model"""
    vec_env = make_vec_env(_make_env_fn(n_final, max_steps), n_envs=n_envs)
    model = DQN("MlpPolicy", vec_env, verbose=1, buffer_size=200_000, learning_starts=5_000,
        batch_size=128, gamma=0.99, learning_rate=1e-4, exploration_fraction=0.3,exploration_final_eps=0.05,
        target_update_interval=1000,train_freq=4)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model

class RLAgent:
    """Wraps a trained DQN policy in a regular agent to include in games and benchmarking."""
    def __init__(self, items, model_path="permutation_puzzle/rl_model.zip"):
        self.items = list(items)
        self._item_to_idx = {item: i for i, item in enumerate(self.items)}
        self.n = len(items)
        self.max_steps = 20
        self._rng = np.random.default_rng()
        # Loading saved model from the path:
        file_name = Path(model_path).stem
        self.model = DQN.load(Path(__file__).parent / file_name)
        self._swap_pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        self._pair_to_action = {pair: idx for idx, pair in enumerate(self._swap_pairs)}
        self.reset()

    def reset(self):
        self.history = []
        self.best_guess = np.random.permutation(self.n)
        self.best_score = -1
        self.steps = 0
        self._first_guess_pending = True
        self.last_action = None
        self.last_delta =0.0
        self.steps_since_improvement = 0
        self._pending_action = None
        self.tried_actions_since_improvement = np.zeros(len(self._swap_pairs), dtype=np.float32)

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

    def _last_action_onehot(self):
        oh = np.zeros(2 * self.n, dtype=np.float32)
        if self.last_action is not None:
            i, j = self.last_action
            oh[i] = 1.0
            oh[self.n + j] = 1.0
        return oh

    def _get_obs(self, max_steps=30):
        comp = self._compatibility_matrix().flatten()
        best = self._best_guess_onehot().flatten()
        best_score_norm = np.array([self.best_score / self.n], dtype=np.float32)
        steps_norm = np.array([self.steps / max_steps], dtype=np.float32)
        last_action = self._last_action_onehot()
        last_delta_norm = np.array([self.last_delta / self.n], dtype=np.float32)
        streak_norm = np.array([min(self.steps_since_improvement, self.max_steps) / self.max_steps], dtype=np.float32)
        return np.concatenate([comp, best, best_score_norm, steps_norm, last_action, last_delta_norm, streak_norm, self.tried_actions_since_improvement]).astype(np.float32)

    def _masked_greedy_action(self, obs):
        # masking against unuseful actions in the latest rounds
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with th.no_grad():
            q_values = self.model.policy.q_net(obs_tensor).cpu().numpy().flatten()

        mask = self.tried_actions_since_improvement.astype(bool)
        if mask.all():
            # fall back when there isn't any left.
            return int(np.argmax(q_values))
        q_values = np.where(mask, -np.inf, q_values)
        return int(np.argmax(q_values))

    # Agent playing similar to the heuristic agents
    def make_guess(self):
        self.steps += 1
        if self._first_guess_pending:
            self._first_guess_pending = False
            guess_idx = self.best_guess
            self._pending_action = None
        else:
            obs = self._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            i, j = self._swap_pairs[int(action)]
            guess_idx = self.best_guess.copy()
            guess_idx[i], guess_idx[j] = guess_idx[j], guess_idx[i]
            self._pending_action = (i,j)
        return [self.items[idx] for idx in guess_idx]

    def receive_feedback(self, guess, score):
        guess_idx = np.array([self.items.index(x) for x in guess])
        self.history.append((guess_idx, score))

        if self._pending_action is not None:
            self.last_action = self._pending_action
            self.last_delta = float(score - self.best_score)

        if score > self.best_score:
            self.best_score = score
            self.best_guess = guess_idx
            self.steps_since_improvement = 0
        elif self._pending_action is not None:
            self.steps_since_improvement += 1
