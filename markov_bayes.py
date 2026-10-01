"""Character-level language model and Naive Bayes classifier for movie scripts.

The script does two things with the text of a movie script:

1. It builds unigram, bigram, and trigram models over 27 characters (the
   lowercase letters and the space) and uses them as a Markov chain to
   generate new text one character at a time.
2. It builds a Naive Bayes classifier that decides which of two scripts a
   piece of text came from, using only character frequencies and a prior
   for each script.

The generated sentences are then run through the classifier to check that it
attributes them to the script they were generated from.
"""

import random
import re
import string
from collections import Counter
from itertools import product

import numpy as np

# The script the language model is trained on, and a second script for the
# classifier to tell it apart from.
SCRIPT_FILE = 'blackpanther.txt'
COMPARISON_FILE = 'script.txt'
PROCESSED_FILE = 'newblackpanther.txt'

# Prior probability that a piece of text comes from each script.
P_script = 0.85
P_comparison = 0.15

# Length of each generated sentence, in characters.
num_characters = 1000

# All possible characters: the space and the 26 lowercase letters.
allchar = " " + string.ascii_lowercase


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def process_text(data):
    """Lowercase the text and keep only letters and single spaces."""
    data = data.lower()
    data = re.sub(r"[^a-z ]+", "", data)
    data = " ".join(data.split())
    data = re.sub(' +', ' ', data)
    return data


def get_unigram_prob(data):
    """Return the share of the text made up by each character."""
    unigram = Counter(data)
    return {ch: round(unigram[ch] / len(data), 4) for ch in allchar}


with open(SCRIPT_FILE, encoding="utf-8") as f:
    data = process_text(f.read())

with open(PROCESSED_FILE, 'w') as f:
    f.write(data)

with open(COMPARISON_FILE, encoding="utf-8") as f:
    comparison_data = process_text(f.read())


# ---------------------------------------------------------------------------
# N-gram models
# ---------------------------------------------------------------------------

def ngram(data, n):
    """Count every sequence of n characters in the text.

    Sequences that never appear are included with a count of 0, so every
    possible n-gram has an entry.
    """
    d = dict.fromkeys(["".join(i) for i in product(allchar, repeat=n)], 0)
    d.update(Counter(data[x: x + n] for x in range(len(data) - n + 1)))
    return d


unigram = Counter(data)
script_unigram_prob = get_unigram_prob(data)
comparison_unigram_prob = get_unigram_prob(comparison_data)

# Bigram: probability of a character given the one before it. The second
# version uses Laplace smoothing, so no transition has probability 0.
bigram = ngram(data, 2)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}
bigram_prob_L = {c: (bigram[c] + 1) / (unigram[c[0]] + len(allchar)) for c in bigram}

# Trigram: probability of a character given the two before it, smoothed.
trigram = ngram(data, 3)
trigram_prob_L = {c: (trigram[c] + 1) / (bigram[c[:2]] + len(allchar)) for c in trigram}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def weighted_choice(collection, weights):
    """Randomly choose an element from collection according to weights.

    Based on https://python-course.eu/numerical-programming/weighted-probabilities.php
    """
    weights = np.array(weights)
    weights_sum = weights.sum()
    weights = weights.cumsum() / weights_sum
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return collection[i]
    return collection[-1]


def gen_bi(c):
    """Generate the next character given the previous one."""
    w = [bigram_prob[c + i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]


def gen_tri(ab):
    """Generate the next character given the previous two."""
    w = [trigram_prob_L[ab + i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]


def gen_sen(c, num):
    """Generate a sentence of num characters that starts with character c.

    Each new character is drawn from the trigram model. If the last two
    characters never appear together in the script, the bigram model is used
    for that step instead.
    """
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res


# One sentence starting with each letter of the alphabet.
sentences = [gen_sen(char, num_characters) for char in string.ascii_lowercase]


# ---------------------------------------------------------------------------
# Naive Bayes classifier
# ---------------------------------------------------------------------------

# Posterior probability that a single character came from the comparison
# script, by Bayes' rule:
#   P(comparison | char) = P(char | comparison) * P(comparison) / P(char)
posterior_probs = []
for char in allchar:
    P_char = (P_script * script_unigram_prob[char]
              + P_comparison * comparison_unigram_prob[char])
    P_comparison_given_char = (comparison_unigram_prob[char] * P_comparison) / P_char
    posterior_probs.append(round(P_comparison_given_char, 4))


def classify(sentence):
    """Return 0 if the sentence is more likely from the script, 1 otherwise.

    Characters are treated as independent given the source. Log probabilities
    are summed rather than multiplying probabilities, which would underflow
    on a sentence this long.
    """
    log_prob_script = np.log(P_script)
    log_prob_comparison = np.log(P_comparison)
    for char in sentence:
        log_prob_script += np.log(script_unigram_prob[char])
        log_prob_comparison += np.log(comparison_unigram_prob[char])
    return 0 if log_prob_script > log_prob_comparison else 1


predictions = [classify(sentence) for sentence in sentences]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_unigram_prob(unigram_prob_dict):
    print(', '.join(f"{unigram_prob_dict[ch]:.4f}" for ch in allchar))


print(f"Unigram probabilities for {SCRIPT_FILE}:")
print_unigram_prob(script_unigram_prob)

# One row per first character, one column per second character.
print("\nBigram transition probabilities, without smoothing:")
for ch1 in allchar:
    print(", ".join(f"{bigram_prob[ch1 + ch2]:.4f}" for ch2 in allchar))

print("\nBigram transition probabilities, with Laplace smoothing:")
for ch1 in allchar:
    print(", ".join(f"{bigram_prob_L[ch1 + ch2]:.4f}" for ch2 in allchar))

print("\nGenerated sentences:")
for letter, sentence in zip(string.ascii_lowercase, sentences):
    print(f"Sentence for letter '{letter}': {sentence}\n")

print(f"Unigram probabilities for {COMPARISON_FILE}:")
print_unigram_prob(comparison_unigram_prob)

print(f"\nPosterior probability of {COMPARISON_FILE} given each character:")
print(', '.join(f"{prob:.4f}" for prob in posterior_probs))

print(f"\nPredicted source of each generated sentence "
      f"(0 = {SCRIPT_FILE}, 1 = {COMPARISON_FILE}):")
print(predictions)
