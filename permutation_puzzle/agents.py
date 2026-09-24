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
        for _ in range(100000):
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
        for _ in range(20000):
            cand = tuple(random.sample(self.items, len(self.items)))
            if cand not in self.tried_guesses and self._is_consistent(cand):
                self.tried_guesses.add(cand)
                return list(cand)
        return random.sample(self.items, len(self.items))

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

        # Uses the systematic agent's global sampling fallback
        attempts = 0
        while len(pool) < target_size and attempts < 30000:
            attempts += 1
            cand_tuple = tuple(random.sample(self.items, len(self.items)))
            # Ensures it is not tried, and consistent with the best guess
            if cand_tuple not in self.tried_guesses and self._is_consistent(cand_tuple):
                pool.append(cand_tuple)

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
                self.tried_guesses.add(tuple(best_candidate))

        return list(best_candidate)

    def receive_feedback(self, guess, score):
        guess_tuple = tuple(guess)
        self.history.append((guess_tuple, score))

        # Update for new best guess
        if score > self.best_score:
            self.best_score = score
            self.best_guess = guess_tuple

class KnuthAgent:
    """"
    Classic Mastermind's minimax strategy (Knuth, 1977): pick the guess
    that minimizes the size of the largest group of still possible
    secrets it could leave behind, across every score it might receive.
    That is: for each candidate guess, partition the remaining possible
    secrets by what score each one would produce against that guess, and
    prefer the guess whose worst case partition is smallest.
    A sampled pool of still possible candidates is kept instead of the full search space.
    """

    def __init__(self, items, pool_size=150):
        self.items = list(items)
        self.pool_size = pool_size
        self.reset()

    def reset(self):
        self.history = []
        self.tried_guesses = set()
        self.best_guess = None
        self.best_score = -1
        self.candidates = None

    def _is_consistent(self, candidate):
        for past_guess, past_score in self.history:
            matches = sum(1 for c, g in zip(candidate, past_guess) if c == g)
            if matches != past_score:
                return False
        return True

    def _bootstrap_candidates(self, target_size):
        """When there is no history yet to filter by, any distinct random permutation
        is a valid starting hypothesis for bootstrapping."""
        seen = set()
        candidates = []
        attempts = 0
        max_attempts = target_size * 50
        while len(candidates) < target_size and attempts < max_attempts:
            attempts += 1
            cand = tuple(random.sample(self.items, len(self.items)))
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)
        return candidates

    def _top_up_candidates(self, target_size):
        """An attempt to grow the pool back toward target size with new candidates consistent with the full history.
        Local perturbations of surviving candidates is genuinely guaranteed consistent with everything so far;
        so small edits of it have much better odds of staying consistent too."""
        seen = set(self.candidates)
        added = []

        if self.candidates:
            attempts = 0
            max_local_attempts = target_size * 50
            while len(self.candidates) + len(added) < target_size and attempts < max_local_attempts:
                attempts += 1
                seed = random.choice(self.candidates)
                cand = list(seed)
                num_swaps = random.randint(1, min(3, len(cand) // 2))
                for _ in range(num_swaps):
                    i, j = random.sample(range(len(cand)), 2)
                    cand[i], cand[j] = cand[j], cand[i]
                cand = tuple(cand)
                if cand not in seen and self._is_consistent(cand):
                    seen.add(cand)
                    added.append(cand)

        attempts = 0
        max_global_attempts = target_size * 50
        while len(self.candidates) + len(added) < target_size and attempts < max_global_attempts:
            attempts += 1
            cand = tuple(random.sample(self.items, len(self.items)))
            if cand not in seen and self._is_consistent(cand):
                seen.add(cand)
                added.append(cand)

        self.candidates.extend(added)

    def _worst_case_partition_size(self, guess, candidates):
        """The worst case remaining search space after this guess, across every outcome.
        ie: If the current guess is played against the candidate pool, how many of
        them could still share the same resulting score"""
        buckets = {}
        for secret in candidates:
            score = sum(1 for g, s in zip(guess, secret) if g == s)
            buckets[score] = buckets.get(score, 0) + 1
        return max(buckets.values())

    def _local_search_fallback(self):
        """Used when the candidate pool has genuinely collapsed to empty;
        A sampled pool might not contain the actual secret so it shrinks toward nothing as constraints accumulate.
        What matters is not discarding everything learned when it happens.
        Falls back to Logic Agent's proven approach (local perturbation of best guess, filtered for consistency)
        rather than a fully blind guess, so the rest of the episode still makes informed progress
        even without a usable minimax candidate pool."""
        if self.best_guess is not None:
            attempts = 0
            while attempts < 20000:
                attempts += 1
                cand = list(self.best_guess)
                num_swaps = random.randint(1, min(3, len(cand) // 2))
                for _ in range(num_swaps):
                    i, j = random.sample(range(len(cand)), 2)
                    cand[i], cand[j] = cand[j], cand[i]
                cand = tuple(cand)
                if cand not in self.tried_guesses and self._is_consistent(cand):
                    return cand

        #Fallback to random sampling
        for _ in range(20000):
            cand = tuple(random.sample(self.items, len(self.items)))
            if cand not in self.tried_guesses:
                return cand
        return tuple(random.sample(self.items, len(self.items)))

    def make_guess(self):
        if self.candidates is None:
            self.candidates = self._bootstrap_candidates(self.pool_size)
        elif len(self.candidates) < self.pool_size:
            self._top_up_candidates(self.pool_size)

        if not self.candidates:
            guess = self._local_search_fallback()
            self.tried_guesses.add(guess)
            return list(guess)

        best_guess = None
        best_worst_case = None
        for guess in self.candidates:
            if guess in self.tried_guesses:
                continue
            worst_case = self._worst_case_partition_size(guess, self.candidates)
            if best_worst_case is None or worst_case < best_worst_case:
                best_worst_case = worst_case
                best_guess = guess

        if best_guess is None:
            # Random fall back to any of the candidates
            best_guess = self.candidates[0]

        # Adding only the guess played to the tried list
        self.tried_guesses.add(best_guess)
        return list(best_guess)

    def receive_feedback(self, guess, score):
        guess_t = tuple(guess)
        self.history.append((guess_t, score))
        if score > self.best_score:
            self.best_score = score
            self.best_guess = guess_t

        if self.candidates is not None:
            # Prune the existing pool based on the new constraints
            self.candidates = [c for c in self.candidates if sum(1 for x, g in zip(c, guess_t) if x == g) == score]

