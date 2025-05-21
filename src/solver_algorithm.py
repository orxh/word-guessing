import numpy as np
import random
import abc


class BaseSolverAlgorithm(abc.ABC):
    """Abstract base class for Contexto solver algorithms."""

    MAX_TURNS = 100
    NUM_STARTER_WORDS = 5

    def __init__(
        self, vocab: list[str], embeddings: np.ndarray, starter_words: set[str]
    ):
        self.vocab = vocab
        self.embeddings = embeddings

        self.word2idx: dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

        self.turn: int = 1
        self.word_ranks: list[tuple[str, int]] = []
        self.used_indices: set[int] = set()

        self.starter_indices = [
            self.word2idx[w] for w in starter_words if w in self.word2idx
        ]
        random.shuffle(self.starter_indices)

    @abc.abstractmethod
    def guess_word(self) -> str:
        """Return the next guess according to the current strategy state."""
        pass

    @abc.abstractmethod
    def update_with_rank(self, word: str, rank: int):
        """Update internal state given the rank returned by the game."""
        pass


class SimilarityRankingAlgorithm(BaseSolverAlgorithm):
    """
    Contexto solver based on similarity ranking to previous guesses.
    """

    NUM_NEIGHBOR_SAMPLES = 200
    COMMON_WORD_THRESHOLD = 2000
    RANK_RATIO = 1.1
    MAX_RANK_LENGTH = 15

    def __init__(
        self, vocab: list[str], embeddings: np.ndarray, starter_words: set[str]
    ):
        super().__init__(vocab, embeddings, starter_words)
        self.word_similarities: dict[str, np.ndarray] = {}

    def guess_word(self) -> str:
        # Use common starter words for the first few moves.
        if self.turn <= self.NUM_STARTER_WORDS and self.starter_indices:
            idx = random.choice(self.starter_indices)
            self.starter_indices.remove(idx)
            word = self.vocab[idx]
            return word

        return self._guess_with_similarity_ranking()

    def update_with_rank(self, word: str, rank: int):
        word_idx = self.word2idx.get(word)

        # If the user changes the guess to a word not in our vocabulary, do nothing.
        if word_idx is None:
            return

        # If the guess is not accepted by the game, do nothing.
        if rank == -1:
            self.used_indices.add(word_idx)
            return

        # If the user changes the guess to an already guessed word, do nothing.
        if word_idx in self.used_indices:
            return

        self.turn += 1
        self.used_indices.add(word_idx)

        self.word_ranks.append((word, rank))
        self.word_ranks.sort(key=lambda x: x[1])
        self._get_or_update_similarities(word)

    def _get_or_update_similarities(self, word: str) -> np.ndarray:
        if word in self.word_similarities:
            return self.word_similarities[word]

        word_idx = self.word2idx[word]
        similarities = self.embeddings @ self.embeddings[word_idx]
        self.word_similarities[word] = similarities
        return similarities

    def _get_target_rank(self) -> list[tuple[str, int]]:
        """
        Construct a target rank based on the previous guesses.
        """
        top_word, last_rank = self.word_ranks[0]
        chosen_word_ranks = [(top_word, last_rank)]
        for i in range(1, len(self.word_ranks)):
            current_word, current_rank = self.word_ranks[i]
            if current_rank / last_rank > self.RANK_RATIO:
                chosen_word_ranks.append((current_word, current_rank))
                last_rank = current_rank

            if len(chosen_word_ranks) >= self.MAX_RANK_LENGTH:
                break

        return chosen_word_ranks

    def _sample_word_indices(self) -> np.ndarray:
        """
        Sample a set of word indices from the vocabulary for the current turn.
        """
        top_word, top_rank = self.word_ranks[0]

        if top_rank > 10:
            sampled_word_indices = list(range(len(self.vocab)))
        else:
            # Sample the nearest neighbors of the current top-ranked word.
            top_word_similarities = self._get_or_update_similarities(top_word)
            sampled_word_indices = np.argsort(top_word_similarities)[-self.NUM_NEIGHBOR_SAMPLES:]

        sampled_word_indices = set(sampled_word_indices) - self.used_indices
        return np.array(list(sampled_word_indices))

    def _get_winner_from_candidates(
        self, candidate_word_indices: np.ndarray, candidate_scores: np.ndarray
    ) -> str:
        """
        Get the winner from the candidate words based on the scores.
        """
        max_score = np.max(candidate_scores)
        best_scores_indices = np.where(candidate_scores == max_score)[0]

        # Prefer common words (idx < COMMON_WORD_THRESHOLD) in case of ties.
        preferred_candidates = []
        other_best_candidates = []

        for score_idx in best_scores_indices:
            word_vocab_idx = candidate_word_indices[score_idx]
            if word_vocab_idx < self.COMMON_WORD_THRESHOLD:
                preferred_candidates.append(score_idx)
            else:
                other_best_candidates.append(score_idx)

        if preferred_candidates:
            chosen_score_idx = np.random.choice(preferred_candidates)
        else:
            chosen_score_idx = np.random.choice(other_best_candidates)

        return self.vocab[candidate_word_indices[chosen_score_idx]]

    def _guess_with_similarity_ranking(self) -> str:
        """
        Calculate scores for candidate words based on how well their similarity profile
        matches the established rank order of previous guesses. Scores are based on Spearman's footrule distance
        between the target rank order and the rank order induced by the candidate's similarities.
        """

        chosen_word_ranks = self._get_target_rank()
        M = len(chosen_word_ranks)
        target_rank = np.arange(M)

        # We can afford to compare whole vocabulary (~50k words).
        # 50k * 50 floats ~= 10MB; guess similarities are cached.
        candidate_word_indices = self._sample_word_indices()
        sims_matrix = np.empty((len(candidate_word_indices), M))
        for guess_idx, (guess_word, _) in enumerate(chosen_word_ranks):
            guess_similarities = self._get_or_update_similarities(guess_word)
            sims_matrix[:, guess_idx] = guess_similarities[candidate_word_indices]

        # For each candidate, get the rank of guessed words ordered by similarity (descending).
        observed_ranks = np.argsort(np.argsort(-sims_matrix, axis=1), axis=1)
        # Compute the Spearman's footrule distance for each candidate.
        candidate_scores = -np.sum(np.abs(observed_ranks - target_rank), axis=1)
        return self._get_winner_from_candidates(
            candidate_word_indices, candidate_scores
        )
