import string
import re
from collections import Counter
from itertools import product
from itertools import permutations
import random
from numpy import cumsum
import numpy as np
from numpy.ma import log

# adjust on your own
# my values were as follows: 0.85, 0.15, 1000
P_my = 0.85
P_fake = 0.15
num_charactors = 1000

# I chose the movie black panther and stored the text in the blackpanther.txt
with open("blackpanther.txt", encoding="utf-8") as f:
    data = f.read()

# This function takes a string `data` as input and preprocesses it for further analysis.
# The following steps are taken:
# Convert the entire text to lowercase to make it case-insensitive.
# Remove all non-alphabetic characters from the text, leaving only letters and spaces.
# Remove any excess spaces between words by splitting the text into words and then joining them with a single space.
# Replace multiple consecutive spaces with a single space to have consistent spacing.

def process_text(data):
    ''' Preprocess the text data '''
    data = data.lower()
    data = re.sub(r"[^a-z ]+", "", data)
    data = " ".join(data.split())
    data = re.sub(' +', ' ', data)

    return data
# Assuming `data` is a string containing the text to be processed,
# the following line calls the `process_text` function and stores
# the preprocessed data in the variable `processed_data`.
processed_data = process_text(data)

# The following block of code opens a file named 'newblackpanther.txt' in write mode ('w').
# It writes the preprocessed text data into the file and saves it.
with open('newblackpanther.txt', 'w') as f:
    f.write(processed_data)
data = process_text(data)

# all possible characters
allchar = " " + string.ascii_lowercase

# unigram
unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch] / len(data), 4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

# to distinguish between fake_unigram_prob below
my_unigram_prob = unigram_prob


# This function generates n-grams from a given input data.
# The following steps must be followed:
# Import the 'product' function from the 'itertools' module,
# which will be used to generate all possible n-length combinations of characters.
# Create a dictionary `d` with all possible n-grams as keys, initialized with a count of 0 for each n-gram.
# 'allchar' is assumed to be a list of all possible characters, and `repeat=n`
# ensures n-length combinations are generated.

def ngram(n):
    ''' Generate n-gram '''
    # all possible n-grams
    d = dict.fromkeys(["".join(i) for i in product(allchar, repeat=n)], 0)
    # Update the counts of each n-gram in the dictionary by iterating over overlapping
    # substrings of length 'n' in the 'data' string.
    # 'Counter' is used to efficiently count the occurrences of each n-gram in the 'data' string.
    # 'data[x : x + n]' slices the 'data' string to extract each n-gram.
    # 'range(len(data) - 1)' ensures that we don't go beyond the length of the 'data' string,
    # so the last n-gram is excluded.
    d.update(Counter(data[x : x + n] for x in range(len(data) - 1)))
    return d

# bigram
# Generate the bigrams from the text data and store them in the 'bigram' dictionary.
bigram = ngram(2)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}
bigram_prob_L = {c: (bigram[c] + 1) / (unigram[c[0]] + 27) for c in bigram}

# trigram
# Generate the trigrams from the text data and store them in the 'trigram' dictionary.
trigram = ngram(3)
trigram_prob_L = {c: (trigram[c] + 1) / (bigram[c[:2]] + 27) for c in trigram}


# based on https://python-course.eu/numerical-programming/weighted-probabilities.php
# The following lines of code follow these steps:
# Convert the 'weights' list to a NumPy array for easier calculations.
# Calculate the sum of all weights to use in the normalization process.
# Normalize the weights by dividing each weight by the sum of all weights and taking cumulative sums.

def weighted_choice(collection, weights):
    """Randomly choose an element from collection according to weights"""
    weights = np.array(weights)
    weights_sum = weights.sum()
    weights = weights.cumsum() / weights_sum
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return collection[i]


# Function to generate the second character of a bigram based on the given first character 'c'.
def gen_bi(c):
    ''' Generate the second char '''
    # Create a list 'w' containing the probabilities of all possible second characters for the bigram 'c + i'.
    # The probabilities are obtained from the 'bigram_prob' dictionary, where keys are bigrams,
    # and values are their respective probabilities.
    w = [bigram_prob[c + i] for i in allchar]
    return weighted_choice(allchar, weights=w)[0]

# Function to generate the third character of a trigram based on the given first two characters 'ab'.
def gen_tri(ab):
    ''' Generate the third char '''
    w = [trigram_prob_L[ab + i] for i in allchar]
    # The selected third character is returned as the generated third character for the trigram.
    return weighted_choice(allchar, weights=w)[0]

# Function to generate a sentence (sequence of characters) based on the given first character 'c'
# and the number of characters 'num'.
def gen_sen(c, num):
    ''' generate the second char'''
    res = c + gen_bi(c)
    # Generate the remaining characters (up to 'num' characters) for the sentence.
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
        # Return the generated sentence.
    return res


# The following lines generate sentences for the project
sentences = []
for char in allchar:
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

## fake script
with open("script.txt", encoding="utf-8") as f:
    data = f.read()

data = process_text(data)

unigram = Counter(data)
unigram_prob = {ch: round(unigram[ch] / len(data), 4) for ch in allchar}
uni_list = [unigram_prob[c] for c in allchar]

fake_unigram_prob = unigram_prob

count = 0
for char in allchar:
    count += 1
    print(
        P_fake
        * fake_unigram_prob[char]
        / (P_fake * fake_unigram_prob[char] + P_my * my_unigram_prob[char])
    )

# print(count)

for sentence in sentences:
    my = log(P_my)
    fake = log(P_fake)
    for char in sentence:
        my += np.log10(my_unigram_prob[char])
        fake += np.log10(fake_unigram_prob[char])
    if my > fake:
        print("0")
    else:
        print("1")


# The following lines are for Question 2
# We start off by defining a function to print
# the unigram probabilities from a given unigram probability dictionary.
def print_unigram_prob(unigram_prob_dict):
    all_chars = sorted(unigram_prob_dict.keys())
    print(', '.join(f"{unigram_prob_dict[ch]:.4f}" for ch in all_chars))
# Print the unigram probabilities for each character on a single line.
# We use a list comprehension to format and join the probabilities with commas.
 # The probabilities are formatted to display four decimal places for better readability.
print("Unigram Probabilities for original script:")
print_unigram_prob(my_unigram_prob)

# The following lines are for Question 3
# Calculate and print bigram transition probabilities
print("Without Soothing:")

# Iterate over each character 'ch1' in 'allchar'.
for ch1 in allchar:
    # Initialize a list 'bigram_probs' to store the probabilities for each bigram starting with 'ch1'.
    # We use a list comprehension to calculate the probabilities for all bigrams with 'ch1' as the starting character.
    bigram_probs = [round(bigram_prob[ch1 +ch2], 4) for ch2 in allchar]

    # Print the probabilities for all bigrams starting with 'ch1' on a single line, separated by commas.
    # The probabilities are formatted to display four decimal places for better readability.
    print(", ".join(f"{prob:.4f}" for prob in bigram_probs))

# The following lines of code are Question 4 on the project.
# We start off by following the steps:
print("With Soothing:")

# Iterate over each character 'char1' in 'allchar'.

for char1 in allchar:
    # Initialize an empty list 'line' to store the probabilities for each bigram starting with 'char1'.
    line = []
    # Iterate over each character 'char2' in 'allchar'.
    for char2 in allchar:
        # Get the pre-calculated probability for the bigram 'char1 + char2' from the 'bigram_prob_L' dictionary.
        # If the bigram is not found in the dictionary, the default value is 0.
        prob = bigram_prob_L.get(char1 + char2, 0)
        line.append("{:.4f}".format(prob))
        # Print the probabilities for all bigrams starting with 'char1' on a single line, separated by commas.
    print(", ".join(line))

# The following lines of code are used for Q5 on the Project
# We start off by initializing an empty list to store the generated sentences.
sentences = []
for char in string.ascii_lowercase:
    # Call the function 'gen_sen' with the current character 'char' and 'num_charactors' as arguments.
    # It generates a sentence based on the given character and number of characters in the sentence.
    # The generated sentence is then appended to the 'sentences' list.
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

# Iterate through the list of generated sentences and print each sentence.
for sentence in sentences:
    print(sentence)


# Question 6
sentences = []

for char in string.ascii_lowercase:
    sentence = gen_sen(char, num_charactors)
    sentences.append(sentence)

for i, sentence in enumerate(sentences):
    print(f"Sentence for letter '{string.ascii_lowercase[i]}': {sentence}\n")

# Question 7
def print_unigram_prob(unigram_prob_dict):
    all_chars = sorted(unigram_prob_dict.keys())
    print(', '.join(f"{unigram_prob_dict[ch]:.4f}" for ch in all_chars))

print("Unigram Probabilities for fake script (Question 2 Answer):")
print_unigram_prob(fake_unigram_prob)

# The following lines of code are used for Question 8 on the project.
# Initialize an empty list to store the posterior probabilities for each character in 'allchar'.

posterior_probs = []

# Iterate over each character in 'allchar'.
for char in allchar:
    # Get the pre-calculated unigram probability of the current character 'char' for the 'my' class.
    # 'my_unigram_prob' is assumed to be a pre-calculated dictionary that maps characters to
    # their respective unigram probabilities for the 'my' class.
    P_my_char = my_unigram_prob[char]

    # Get the pre-calculated unigram probability of the current character 'char' for the 'fake' class.
    # 'fake_unigram_prob' is assumed to be a pre-calculated dictionary that maps characters to their
    # respective unigram probabilities for the 'fake' class.
    P_fake_char = fake_unigram_prob[char]

    # Calculate the overall probability of encountering the current character 'char' in the entire dataset.
    # 'P_my' is the pre-calculated probability of the 'my' class, and 'P_fake'
    # is the pre-calculated probability of the 'fake' class.
    P_char = P_my * P_my_char + P_fake * P_fake_char

    # Calculate the posterior probability of the 'fake' class given the presence of the current character 'char'.
    # This is computed using Bayes' theorem, where we calculate the probability of 'fake' given 'char'.
    # The formula is: P(fake|char) = (P(fake_char) * P(fake)) / P(char)
    P_fake_given_char = (P_fake_char * P_fake) / P_char

    # Append the rounded value of P_fake_given_char (posterior probability) to the 'posterior_probs' list.
    posterior_probs.append(round(P_fake_given_char, 4))

print("Unigram Probabilities for fake script:")
print(', '.join(f"{prob:.4f}" for prob in posterior_probs))

# The following lines of code are used in Question 9 of the project.
# We start off by following the steps:
# Initialize an empty list to store the predictions for each sentence.
# Iterate over each sentence in the 'sentences' list.
predictions = []
for sentence in sentences:

    # Calculate the log probability of the 'my' class (e.g., the probability of a sentence being associated with 'my').
    # 'P_my' is assumed to be a pre-calculated probability value for the 'my' class.
    # Calculate the log probability of the 'fake' class (e.g., the probability of a
    # sentence being associated with 'fake').
    # 'P_fake' is assumed to be a pre-calculated probability value for the 'fake' class.
    log_prob_my = np.log(P_my)
    log_prob_fake = np.log(P_fake)

    # Iterate over each character in the current 'sentence'.
    # For each character, update the log probability of the 'my' class by adding the log probability
    # of the character based on the 'my_unigram_prob'.
    # 'my_unigram_prob' is assumed to be a pre-calculated dictionary that maps
    # characters to their respective unigram probabilities for the 'my' class.

    for char in sentence:
        log_prob_my += np.log(my_unigram_prob[char])
        log_prob_fake += np.log(fake_unigram_prob[char])

    # After iterating through all characters in the sentence,
    # compare the final log probabilities for the 'my' and 'fake' classes.
    # If the log probability of the 'my' class is greater than that of the 'fake' class,
    # predict the sentence belongs to the 'my' class (0).
    # Otherwise, predict the sentence belongs to the 'fake' class (1).
    predictions.append(0 if log_prob_my > log_prob_fake else 1)

# Print the list of predictions, where each element represents the predicted class for the corresponding sentence.
print(predictions)
