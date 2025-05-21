import numpy as np
from solver_algorithm import SimilarityRankingAlgorithm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLOVE_WORD_LIST_PATH = "data/glove.42B.300d.clean.txt"

COMMON_STARTER_WORDS = {
    "time",
    "person",
    "year",
    "day",
    "thing",
    "world",
    "life",
    "hand",
    "part",
    "eye",
    "place",
    "work",
    "week",
    "point",
    "government",
    "company",
    "number",
    "group",
    "problem",
    "fact",
}


def _load_embeddings(path: str = GLOVE_WORD_LIST_PATH) -> tuple[list[str], np.ndarray]:
    """Load GloVe embeddings.

    Returns
    -------
    vocab : list[str]
        Tokens from the GloVe embeddings.
    vecs  : np.ndarray, shape (|V|, 300)
        L2-normalised vectors in the same order as `vocab`.
    """
    logger.info("Loading GloVe embeddings from %s …", path)
    vocab = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array([float(val) for val in values[1:]], dtype=np.float32)

            norm = np.linalg.norm(vector)
            if norm > 0:
                vocab.append(word)
                vectors.append(vector / norm)

    vecs = np.vstack(vectors).astype(np.float32)
    logger.info(f"Loaded {len(vocab):,} GloVe vectors.")
    return vocab, vecs


def main():
    print("\nContexto Solver\n———————————————")
    print("Enter the rank returned by the game after each guess (1 = correct).")
    print("Enter 'u' if the game does not recognise the word, or 'q' to quit.\n")
    vocab, vecs = _load_embeddings()

    solver = SimilarityRankingAlgorithm(vocab, vecs, COMMON_STARTER_WORDS)

    while solver.turn < solver.MAX_TURNS:
        guess = solver.guess_word()
        print(f"\nGuess #{solver.turn}: {guess}")

        rank_input = input("Rank: ").strip().lower()
        if rank_input == "q":
            print("Exiting - goodbye!")
            break
        if rank_input == "u":
            # The word is not recognized by the game.
            solver.update_with_rank(guess, -1)
            print(f'Marked "{guess}" as unrecognized by the game for the solver.')
            continue
        try:
            rank = int(rank_input)
            if rank <= 0:
                print("Rank must be a positive integer - try again.")
                continue
        except ValueError:
            print("Invalid input - please enter a number, 'u', or 'q'.")
            continue

        if rank == 1:
            print(f'Solved with "{guess}" on turn {solver.turn}!')
            break

        word_to_update = guess
        confirmed_word_input = (
            input(
                f'The game processed "{guess}". Did it use a different form (e.g., "go" for "going")?\n'
                "If so, enter the word used by the game. Otherwise, just press Enter: "
            )
            .strip()
            .lower()
        )

        if confirmed_word_input and confirmed_word_input != guess:
            print(
                f'Game used "{confirmed_word_input}". Original guess "{guess}" will be treated as unknown by the solver.'
            )
            solver.update_with_rank(guess, -1)
            word_to_update = confirmed_word_input

        solver.update_with_rank(word_to_update, rank)

    else:
        print("Reached maximum number of turns: stopping.")


if __name__ == "__main__":
    main()
