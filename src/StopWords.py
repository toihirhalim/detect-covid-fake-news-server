from nltk.corpus import stopwords


def remove_stop_words_and_numerics(tokens, language):
    sw = {}
    try:
        sw = stopwords.words(language)
    except:
        sw = {}

    clean_tokens = tokens[:]
    for token in tokens:
        if token in sw or token.isnumeric():
            clean_tokens.remove(token)

    return clean_tokens
