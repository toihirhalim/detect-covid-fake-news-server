import strawberry
from Scraping import get_text
from Tokenize import tokenize_by_words, tokenize_by_sentences
from StopWords import remove_stop_words_and_numerics
from Stemming import stemmer_porter, stemmer_lancaster, isri_lancaster
from Bag_of_words import get_bag_of_words
from lemmatizing import lemmatize
from words_pos_tag import get_list_pos_tag
from SentimentAnalysis import get_sentiment
from FakeNews import analyse
from typing import List


@strawberry.type
class SentimentResultType:
    neg: float
    neu: float
    pos: float
    compound: float


@strawberry.type
class Query:
    @strawberry.field
    def scraping(self, url: str) -> str:
        return get_text(url)

    @strawberry.field
    def tokenize(self, text: str, language: str = "arabic", words: bool = True, stop_words: bool = True) -> List[str]:
        if words:
            tokens = tokenize_by_words(text)
            if stop_words:
                return remove_stop_words_and_numerics(tokens, language)
            return tokens

        return tokenize_by_sentences(text)

    @strawberry.field
    def stemming(self, text: str, language: str = "arabic", lancaster: bool = True, stop_words: bool = True) -> List[List[str]]:
        tokens = tokenize_by_words(text)
        if stop_words:
            tokens = remove_stop_words_and_numerics(tokens, language)
        if language == "arabic":
            return isri_lancaster(tokens)
        if lancaster:
            return stemmer_lancaster(tokens)
        return stemmer_porter(tokens)

    @strawberry.field
    def lemmatizing(self, text: str, language: str = "english", stop_words: bool = True) -> List[List[str]]:
        tokens = tokenize_by_words(text)
        if stop_words:
            tokens = remove_stop_words_and_numerics(tokens, language)

        return lemmatize(tokens)

    @strawberry.field
    def pos_tag(self, text: str, language: str = "english", stop_words: bool = True) -> List[List[str]]:
        tokens = tokenize_by_words(text)
        if stop_words:
            tokens = remove_stop_words_and_numerics(tokens, language)
        return get_list_pos_tag(tokens)

    @strawberry.field
    def bag_of_words(self, text: str, language: str = "arabic", stop_words: bool = True) -> List[str]:
        sentences = tokenize_by_sentences(text)

        if stop_words:
            clean_sentences = []
            for sentence in sentences:
                clean_sentences.append(' '.join(remove_stop_words_and_numerics(tokenize_by_words(sentence), language)))
            sentences = clean_sentences

        return get_bag_of_words(sentences)

    @strawberry.field
    def sentiment(self, text: str = "") -> SentimentResultType:
        result = get_sentiment(text)
        return SentimentResultType(neg=result['neg'], neu=result['neu'], pos=result['pos'], compound=result['compound'])

    @strawberry.field
    def fake_news(self, text: str = "") -> str:
        return analyse(text)


schema = strawberry.Schema(query=Query)
