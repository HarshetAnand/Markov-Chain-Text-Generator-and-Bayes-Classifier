# Markov Chain Text Generator and Naive Bayes Classifier

A from-scratch NLP project implementing a Markov chain language model and a Naive Bayes classifier. The Markov chain generates new sentences based on character-level probability distributions from a source text, while the classifier distinguishes between authentic and "fake" text based on character-level features.

## Features

- Character-level Markov chain text generation
- Unigram, bigram, and trigram probability models
- Laplace smoothing for bigram and trigram probabilities
- Weighted random character selection based on probability distributions
- Naive Bayes classifier for binary text classification
- Bayesian posterior probability calculations
- Sentence generation starting from any character

## Tech Stack

- Python
- NumPy
- Standard library (collections, itertools, re, random, string)

## Data

The script expects two plain-text files in the project folder:

- `source.txt`: the text the language model is trained on, such as a movie script.
- `comparison.txt`: a second text the classifier is trained against.

The script saves the cleaned source text to `source_processed.txt`, generates one 1,000-character sentence for each starting letter, and classifies each one as coming from the source or the comparison text.

## Implementation Details

**Markov Chain Generation:**

- Preprocesses text (lowercase, alphabetic only, normalized spacing)
- Builds unigram, bigram, and trigram probability tables
- Generates new characters based on previous 1-2 characters using weighted random selection
- Falls back to the bigram model when the previous two characters never appear together in the source text

**Naive Bayes Classification:**

- Compares character-level probabilities between two source texts
- Calculates log probabilities to avoid floating point underflow
- Uses Bayes' theorem to compute posterior probabilities
- Classifies generated sentences as belonging to source or "fake" text

## Key Concepts Demonstrated

- N-gram language models
- Markov chain probability generation
- Laplace smoothing for probability estimation
- Naive Bayes classification
- Bayesian inference and posterior probability
- Log probability arithmetic for numerical stability
- Text preprocessing and feature engineering
