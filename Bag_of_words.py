from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()


def get_bag_of_words(corpus):
    X = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names()
