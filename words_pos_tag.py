from nltk import pos_tag
from nltk.corpus import wordnet


def get_pos_tag(tokens):
    return pos_tag(tokens)


def get_list_pos_tag(tokens):
    result = []
    for token in tokens:
        result.append([token, pos_tag([token])[0][1]])
    return result


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None