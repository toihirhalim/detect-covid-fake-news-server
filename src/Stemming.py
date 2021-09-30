from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.isri import ISRIStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()
isri = ISRIStemmer()


def stemmer_porter(tokens):
    result = []
    for word in tokens:
        result.append([word, porter.stem(word)])
    return result


def stemmer_lancaster(tokens):
    result = []
    for word in tokens:
        result.append([word, lancaster.stem(word)])
    return result


def isri_lancaster(tokens):
    result = []
    for word in tokens:
        result.append([word, isri.stem(word)])
    return result
