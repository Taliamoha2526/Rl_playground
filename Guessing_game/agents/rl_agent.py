import numpy as np
from Guessing_game.agents.base_agent import BaseAgent
from Guessing_game.environments.gym_env import GuessWhoEnv
from stable_baselines3 import DQN

class RLAgent(BaseAgent):

    def __init__(self, search_space, model_path=None, max_steps=20):
        super().__init__(search_space, max_steps)
        self.gym_env = GuessWhoEnv(search_space, max_steps=max_steps)
        self.model = None
        # Initialize environment and load if there is existing agent.
        if model_path:
            self.model = DQN.load(model_path)

    @classmethod
    def train_model(cls, characters_df, total_timesteps=150_000,save_path="models/dqn_guess_who.zip" ):
        print(f"\n--- Training Feature-Ratio RL Agent ({total_timesteps} timesteps) ---")
        env = GuessWhoEnv(characters_df, max_steps=100)

        model = DQN("MultiInputPolicy", env, learning_rate=5e-4, buffer_size=50_000,
            learning_starts=1_000, batch_size=128, gamma=0.95,  exploration_fraction=0.4,
            exploration_final_eps=0.02,verbose=0,)

        model.learn(total_timesteps=total_timesteps)
        model.save(save_path)
        print(f"\nModel training complete! Saved to '{save_path}'.\n")
        return save_path

    def take_turn(self):
        if self.model is None:
            return "error", "RL Model not loaded."

        # Synchronize environment tracking masks with actual simulation state
        for idx, candidate_name in enumerate(self.gym_env.df.index):
            if candidate_name in self.rejected or candidate_name not in self.search_space.index:
                self.gym_env.candidate_mask[idx] = 0
            else:
                self.gym_env.candidate_mask[idx] = 1

        for idx, (f, v) in enumerate(self.gym_env.questions):
            if (f, v) in self.history:
                self.gym_env.asked_mask[idx] = 1

        obs = self.gym_env._get_obs()

        # Predict action based on the observations
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        rem_candidates = self.search_space[~self.search_space.index.isin(self.rejected)]
        rem_count = len(rem_candidates)

        # End game trigger
        if rem_count == 1 or (self.step_counter >= self.max_steps - 1 and rem_count <= 5):
            guess_character = rem_candidates.iloc[0]
            return "guess", guess_character

        # Masking to force question selection while candidates > 1
        if rem_count > 1 and action >= self.gym_env.num_questions:
            unasked = [i for i in range(self.gym_env.num_questions) if self.gym_env.asked_mask[i] == 0]
            if unasked:
                # Pick unasked question with split ratio closest to 0.5
                ratios = obs["split_ratios"]
                unasked_ratios = {i: abs(0.5 - ratios[i]) for i in unasked}
                action = min(unasked_ratios, key=unasked_ratios.get)

        if action < self.gym_env.num_questions:
            feature, value = self.gym_env.questions[action]
            return "ask", feature, value
        else:
            guess_idx = action - self.gym_env.num_questions
            guess_character = self.gym_env.df.iloc[guess_idx]
            return "guess", guess_character