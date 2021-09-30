from nltk import word_tokenize
from nltk import sent_tokenize


def tokenize_by_words(text):
    words = word_tokenize(text)
    tokens = [word for word in words if word.isalnum()]

    return tokens


def tokenize_by_sentences(text):
    return sent_tokenize(text)
