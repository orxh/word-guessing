# Contexto Solver

## Environment Setup
* conda env create -f environment.yml
* conda activate word-guesser
* If you only want to run the solver, you can skip the conda setup since only numpy is required.

## Data Setup
* You can skip this step as I am providing the preprocessed data in the `data` folder.
* Download glove vectors from [here](https://nlp.stanford.edu/projects/glove/). I am using the 42B 300d vectors.
* Unzip the file and place the `glove.42B.300d.txt` file in the `data` folder.
* Run `python src/preprocess.py` to preprocess the data. Preprocessing applies the following filters:
    *  Only words that are lowercased are considered.
    *  Only words with length between 3 and 15 characters are considered.
    *  Words that contain non-alphabetic characters are discarded.
    *  Words that are stop words in English are discarded.
    *  Words that are not in the WordNet lexical database are discarded.

## Solver
* Run `python src/contexto_solver.py`

## Strategy
The solver employs a strategy based on word embedding similarities to narrow down the search space for the secret word.

1.  **Initial Guesses**:
    *   The algorithm starts by guessing a predefined number of common starter words.

2.  **Similarity Ranking**:
    *   After the starter words, the algorithm maintains a list of all words guessed so far, along with the rank given by the game for each.
    *   It uses pre-trained GloVe word embeddings.

3.  **Forming a "Target Rank Profile"**:
    *   From the list of guessed words, the algorithm selects a subset of the best-ranked words. This "target rank profile" includes the top-ranked word and other subsequent words where the rank increases by a certain threshold compared to the previous word.

4.  **Generating and Scoring Candidate Words**:
    *   **Candidate Pool**:
        *   If the current best-ranked word is close (low rank number), the candidate pool is narrowed to words semantically similar (nearest neighbors in the embedding space) to this best-ranked word.
        *   Otherwise, the candidate pool is the entire vocabulary. 
    *   **Scoring**: Each candidate word is scored based on how well its pattern of similarities to the words in the "target rank profile" matches the actual rank order of those target words.
        *   For a candidate, cosine similarities are computed with each word in the target rank profile.
        *   Each candidate is scored by using the Spearman's footrule distance between the target rank order and the rank order induced by the candidate's similarities to the target words.
        *  Example:
            *  Target rank profile: [1, 2, 3, 4, 5]
            *  Candidate similarities: [0.9, 0.7, 0.8, 0.6, 0.5]
            *  Candidate's rank order: [1, 3, 2, 4, 5]
            *  Score = -(|1-1| + |2-3| + |3-2| + |4-4| + |5-5|) = -2

5.  **Making a Guess**:
    *   The candidate word with the highest score is chosen as the next guess.
    *   In case of ties, the algorithm prefers more common words (determined by their frequency in the GloVe vocabulary).

6.  **Updating State**:
    *   The game's feedback (rank of the guessed word, or if the word was unrecognized) is used to update the solver's internal state: the list of guessed words and their ranks, and the set of used words.
