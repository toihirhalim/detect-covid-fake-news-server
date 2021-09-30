import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv('data/corona_fake.csv')

df = df.fillna('')
df['title_text_source'] = df['title'] + ' ' + df['text'] + ' ' + df['source']
df = df[df['label'] != '']
df.loc[df['label'] == 'fake', 'label'] = 'FAKE'
df.loc[df['label'] == 'Fake', 'label'] = 'FAKE'
no_of_fakes = df.loc[df['label'] == 'FAKE'].count()[0]
no_of_trues = df.loc[df['label'] == 'TRUE'].count()[0]

stop_words = set(stopwords.words('english'))


def clean(text):
    # Lowering letters
    text = text.lower()

    # Removing html tags
    text = re.sub(r'<[^>]*>', '', text)

    # Removing twitter usernames
    text = re.sub(r'@[A-Za-z0-9]+', '', text)

    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]', '', text)

    # Removing numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    word_tokens = word_tokenize(text)

    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)

    # Joining words
    text = (' '.join(filtered_sentence))
    return text


df['title_text_source'] = df['title_text_source'].apply(clean)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['title_text_source'].values)
X = X.toarray()

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=11)

clf = MultinomialNB()
clf.fit(X_train, y_train)


def analyse(sentence):
    sentence = clean(sentence)
    vectorized_sentence = vectorizer.transform([sentence]).toarray()
    result = clf.predict(vectorized_sentence)
    return result[0]
