import csv

import numpy as np
import re
import itertools
from collections import Counter

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk import RegexpTokenizer


# def load_data_and_labels():
#     """
#     Loads polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     regex = RegexpTokenizer(r"\b[^\d\W]+\b")
#     # Load data from files
#     reader = csv.reader(open("./data/trainset.txt"), delimiter='\t')
#
#     #positive_examples = list(open("./data/trainset.txt", "r").readlines())
#     text = []
#     label = []
#     for row in reader:
#         #print(row[2])
#         text.append(row[2])
#         label.append(row[1])
#
#     label_encoder = LabelEncoder()
#     integer_encoded = label_encoder.fit_transform(label)
#     print(integer_encoded)
#     label = to_categorical(integer_encoded)
#     print(label)
#
#
#     # Split by words
#
#     x_text = [regex.tokenize(s) for s in text]
#     # Generate labels
#
#     return [x_text, label]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    vocabulary["#UNKOWN"] = len(vocabulary.keys())+1
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] if word in vocabulary else "#UNKOWN" for word in sentence] for sentence in sentences])
    return x


def load_data(x,y,tokenizer):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences = [tokenizer(t) for t in x]

    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    y = np.array(y)

    return [x, y, vocabulary, vocabulary_inv]

def load_test_data(x,vocabulary,tokenizer):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences = [tokenizer(t) for t in x]

    sentences_padded = pad_sentences(sentences)
    x = build_input_data(sentences_padded, vocabulary)
    return x
