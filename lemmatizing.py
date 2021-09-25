from nltk.stem import WordNetLemmatizer
from words_pos_tag import nltk_tag_to_wordnet_tag, get_pos_tag


lemmatizer = WordNetLemmatizer()


def lemmatize(tokens):
    nltk_tagged = get_pos_tag(tokens)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    result = []

    for word, tag in wordnet_tagged:
        if tag is None:
            result.append([word, word])
        else:
            result.append([word, lemmatizer.lemmatize(word, tag)])

    return result
