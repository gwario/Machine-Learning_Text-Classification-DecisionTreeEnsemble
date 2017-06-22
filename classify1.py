import nltk
from nltk import word_tokenize
import numpy as np
import csv
import argparse
import string
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

import matplotlib.pyplot as plt

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


# Source: http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
class NLTKPreprocessor(BaseEstimator, TransformerMixin):

	def __init__(self, stopwords=None, punct=None,
	             lower=True, strip=True):
	    self.lower      = lower
	    self.strip      = strip
	    self.stopwords  = stopwords or set(sw.words('english'))
	    self.punct      = punct or set(string.punctuation)
	    self.lemmatizer = WordNetLemmatizer()

	def fit(self, X, y=None):
	    return self

	def inverse_transform(self, X):
	    return [" ".join(doc) for doc in X]

	def transform(self, X):
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

	            # If punctuation, ignore token and continue
	            if all(char in self.punct for char in token):
	                continue

	            # Lemmatize the token and yield
	            lemma = self.lemmatize(token, tag)
	            yield lemma

	def lemmatize(self, token, tag):
	    tag = {
	        'N': wn.NOUN,
	        'V': wn.VERB,
	        'R': wn.ADV,
	        'J': wn.ADJ
	    }.get(tag[0], wn.NOUN)

	    return self.lemmatizer.lemmatize(token, tag)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program.')
    parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'), help='Training data file.')
    # parser.add_argument('--load', metavar='FILE', type=argparse.FileType('r'), help='Model file.')
    parser.add_argument('--classify', metavar='FILE', nargs=1, type=argparse.FileType('r'), help='Data to be classified.')
    parser.print_help()

    args = parser.parse_args()

    classification_data = []
    next(args.classify[0])
    for parts in csv.reader(args.classify[0], delimiter=','):
        #idx = parts[0]
        #title = parts[1]
        #abstract = parts[2]
        classification_data.append(parts)

    classification_data = np.array(classification_data)

    training_data = []
    next(args.train)
    for parts in csv.reader(args.train, delimiter=','):

        #idx = parts[0]
        #title = parts[1]
        #abstract = parts[2]
        #class = parts[3]
        training_data.append(parts)

    training_data = np.array(training_data)
    # tokens = word_tokenize(training_data[0, 2])
    # words = [w.lower() for w in tokens]
    # vocab = sorted(words)
    # frqDst = nltk.FreqDist(words)
    # print(frqDst.most_common(50))

    model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(
                tokenizer=identity, preprocessor=None, lowercase=False
            )),
            ('classifier', RandomForestClassifier(max_depth=8, n_estimators=10, random_state = 0)),
        ])

    training_size = 0.8

    # print("shape: " + str(training_data.shape))

    X = training_data[training_size * training_data.shape[0]:, 2]
    y = training_data[training_size * training_data.shape[0]:, 3]

    # print("y before: " + str(y))

    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # print("y after: " + str(y))

    model.fit(X, y)
	
    # ----- Feature importance diagram -----
    importances = model.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0) 
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # ------ End ------

    test_data = training_data[-((1 - training_size) * training_data.shape[0]):, 2]
    test_data_ground_truth = training_data[-((1 - training_size) * training_data.shape[0]):, 3]
    test_data_ground_truth = labels.fit_transform(test_data_ground_truth)
    result = model.predict(test_data)

    print("predicted labels: " + str(result))
    print("real labels: " + str(test_data_ground_truth))

    print("accuracy: " + str(np.mean(result == test_data_ground_truth)))

    # for idx, val in enumerate(classification_data[:, 0]):
    #     print("{}, {}".format(val, result[idx]))
