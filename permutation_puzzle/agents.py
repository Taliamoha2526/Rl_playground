import random

class RandomAgent:

    def __init__(self, items):
        self.items = list(items)
        self.reset()

    def reset(self):
        self.tried_guesses = set()

    def make_guess(self):
        while True:
            # Randomly choose new order
            guess = tuple(random.sample(self.items, len(self.items)))
            if guess not in self.tried_guesses:
                self.tried_guesses.add(guess)
                return list(guess)

    def receive_feedback(self, guess, score):
        pass


class SystematicAgent:

    def __init__(self, items):
        self.items = list(items)
        self.reset()

    def reset(self):
        self.history = []
        self.tried_guesses = set()

    def _is_consistent(self, candidate):
        #Find matching values with prev guess
        for past_guess, past_score in self.history:
            matches = sum(1 for c, g in zip(candidate, past_guess) if c == g)
            # If they have the same score, the match indicates these elements were correct
            if matches != past_score:
                return False
        return True

    def make_guess(self):
        max_attempts = 100000
        attempts = 0

        #Cue to keep playing
        while attempts < max_attempts:
            cand = tuple(random.sample(self.items, len(self.items)))
            attempts += 1
            #Compare the current candidate with history and matches
            if cand not in self.tried_guesses and self._is_consistent(cand):
                self.tried_guesses.add(cand)
                return list(cand)

        # Fallback if consistency search exceeds attempt threshold
        return random.sample(self.items, len(self.items))

    def receive_feedback(self, guess, score):
        self.history.append((tuple(guess), score))


class LogicAgent:

    def __init__(self, items):
        self.items = list(items)
        self.reset()

    def reset(self):
        self.history = []
        self.current_best = random.sample(self.items, len(self.items))
        self.best_score = -1
        self.tried_guesses = set()

    def _is_consistent(self, candidate):
        #Check the match with previous elements
        for past_guess, past_score in self.history:
            matches = sum(1 for c, g in zip(candidate, past_guess) if c == g)
            # If the score is same, it is likely the elements were correct
            if matches != past_score:
                return False
        return True

    def make_guess(self):
        if not self.history:
            guess = tuple(self.current_best)
            self.tried_guesses.add(guess)
            return list(guess)

        # Swap on the best guess as long as it maintains consistency with the history
        for _ in range(50000):
            cand = list(self.current_best)
            # Perform 1 to 3 up to consistency
            num_swaps = random.randint(1, min(3, len(self.items) // 2))
            for _ in range(num_swaps):
                i, j = random.sample(range(len(cand)), 2)
                cand[i], cand[j] = cand[j], cand[i]

            cand_tuple = tuple(cand)
            if cand_tuple not in self.tried_guesses and self._is_consistent(cand_tuple):
                self.tried_guesses.add(cand_tuple)
                return cand

        # Use random candidate if the local swaps fail
        while True:
            cand = tuple(random.sample(self.items, len(self.items)))
            if cand not in self.tried_guesses and self._is_consistent(cand):
                self.tried_guesses.add(cand)
                return list(cand)

    def receive_feedback(self, guess, score):
        self.history.append((tuple(guess), score))
        # Update the highest score for reference and comparison
        if score > self.best_score:
            self.best_score = score
            self.current_best = guess[:]


class HybridAgent:
    def __init__(self, items, pool_sample_size=100):
        self.items = list(items)
        self.pool_sample_size = pool_sample_size
        self.reset()

    def reset(self):
        self.history = []            # Stores (guess_tuple, score)
        self.tried_guesses = set()
        self.best_guess = None
        self.best_score = -1

    def _is_consistent(self, candidate):
        # Evaluates current candidate match against history
        for past_guess, past_score in self.history:
            matches = sum(1 for c, g in zip(candidate, past_guess) if c == g)
            if matches != past_score:
                return False
        return True

    def _generate_candidate_pool(self, target_size):
        pool = []
        attempts = 0

        # Uses the logic agent to perform swaps
        if self.best_guess is not None:
            while len(pool) < target_size and attempts < 20000:
                attempts += 1
                cand = list(self.best_guess)
                # Perform 1 to 3 swaps
                num_swaps = random.randint(1, min(3, max(2, len(self.items) // 2)))
                for _ in range(num_swaps):
                    i, j = random.sample(range(len(cand)), 2)
                    cand[i], cand[j] = cand[j], cand[i]
                # Matches consistency before using the guess
                cand_tuple = tuple(cand)
                if cand_tuple not in self.tried_guesses and self._is_consistent(cand_tuple):
                    pool.append(cand_tuple)
                    self.tried_guesses.add(cand_tuple)

        # Uses the systematic agent's global sampling fallback
        attempts = 0
        while len(pool) < target_size and attempts < 30000:
            attempts += 1
            cand_tuple = tuple(random.sample(self.items, len(self.items)))
            # Ensures it is not tried, and consistent with the best guess
            if cand_tuple not in self.tried_guesses and self._is_consistent(cand_tuple):
                pool.append(cand_tuple)
                self.tried_guesses.add(cand_tuple)

        return pool

    def make_guess(self):
        # Starts off with random guess
        if not self.history:
            guess = tuple(random.sample(self.items, len(self.items)))
            self.tried_guesses.add(guess)
            self.best_guess = guess
            return list(guess)

        # Generate a candidate pool using the combined approach
        candidate_pool = self._generate_candidate_pool(self.pool_sample_size)

        if not candidate_pool:
            # Fallback use the random guess
            cand = list(self.best_guess)
            i, j = random.sample(range(len(cand)), 2)
            cand[i], cand[j] = cand[j], cand[i]
            return cand

        # Use an entropy metric
        best_candidate = candidate_pool[0]
        max_entropy = -1.0

        # Evaluate potential entropy for a candidate subset
        for cand in candidate_pool[:20]:
            score_counts = {}
            for target in candidate_pool:
                simulated_score = sum(1 for c, t in zip(cand, target) if c == t)
                score_counts[simulated_score] = score_counts.get(simulated_score, 0) + 1
            # Update best guess based on the entropy
            entropy = len(score_counts)
            if entropy > max_entropy:
                max_entropy = entropy
                best_candidate = cand

        return list(best_candidate)

    def receive_feedback(self, guess, score):
        guess_tuple = tuple(guess)
        self.history.append((guess_tuple, score))

        # Update for new best guess
        if score > self.best_score:
            self.best_score = score
            self.best_guess = guess_tuple