#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.


'''

# Python 2/3 compatibility
from __future__ import print_function
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program.')
    parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'), help='Training data file.')
    parser.add_argument('--load', metavar='FILE', type=argparse.FileType('r'), help='Model file.')
    parser.add_argument('classify', metavar='FILE', nargs=1, type=argparse.FileType('r'), help='Data to be classified.')
    parser.print_help()

    args = parser.parse_args()

    print(args)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(max_depth=8, n_estimators=10)),
    ])

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
        #clazz = parts[3]
        training_data.append(parts)

    training_data = np.array(training_data)


    text_clf.fit(training_data[:, 1], training_data[:, 3])
    result = text_clf.predict(classification_data[:, 2])

    print(result)

    for idx, val in enumerate(classification_data[:, 0]):
        print("{}, {}".format(val, result[idx]))

