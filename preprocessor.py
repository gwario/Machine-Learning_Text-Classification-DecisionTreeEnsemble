import logging as log
import string
import sys
import pprint

from nltk import WordNetLemmatizer
from nltk import pos_tag
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from sklearn.base import BaseEstimator, TransformerMixin

if sys.version_info >= (3, 0):
    import importlib

    importlib.reload(sys)
else:
    reload(sys)
    sys.setdefaultencoding('utf8')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s, errors: str(s)


# Source: http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        log.debug("Tokenizing documents (this may take a while...)")
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        document = unicode(document, errors='replace')
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If digit, continue
                if is_number(token):
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                #print(token, tag)
                tag = {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)

                try:
                    synsets = wn.synset(lemma+"."+tag+".01")
                    synonyms = synsets.lemmas()
                    if len(synonyms) > 0:
                        synonym = synonyms[0].name()

                        for token, tag in pos_tag(wordpunct_tokenize(synonym)):
                            # Apply preprocessing to the token
                            token = token.lower() if self.lower else token
                            token = token.strip() if self.strip else token
                            token = token.strip('_') if self.strip else token
                            token = token.strip('*') if self.strip else token

                            # If stopword, ignore token and continue
                            if token in self.stopwords:
                                continue

                            # If digit, continue
                            if is_number(token):
                                continue

                            # If punctuation, ignore token and continue
                            if all(char in self.punct for char in token):
                                continue

                            if token != lemma:
                                print("Synonym for "+lemma+"."+tag+".01:"+token)

                            yield synonym
                    else:
                        yield lemma
                except WordNetError:
                    yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


def additional_data_tokenizer(value):
    """Returns the separated keywords/terms."""
    if isinstance(value, str):
        return value.split(';')
    else:
        return []
