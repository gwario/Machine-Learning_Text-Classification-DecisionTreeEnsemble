#!/usr/bin/env python
"""
Classifies, train and saves the model.

Modes of operation:

Train & classify: (--train --predict)
    Trains the classifier, saves the model to disk and classifies.
Load model & classify: (--model --predict)
    Loads the model and classifies.
Train only: (--train)
    Trains the classifier and saves the model to disk.
"""

# Python 2/3 compatibility
from __future__ import print_function

__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


import logging as log
from sys import exit
import argparse
from pprint import pprint
from datetime import datetime
import pandas as pd

from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import HashingVectorizer
from extractor import Printer, ItemSelector

# Display progress logs on stdout
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')



# This parameters grid is used to find the best parameter configuration for the
# pipeline when --grid was specified.
#TODO add some variation of the default parameters
pipeline_parameters_grid = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__n_iter': (10, 50, 80),
    'clf__max_depth': (2, 5),
    'clf__n_estimators': (10, 80),
}

# This custom set of parameters is used when --grid was NOT specified.
#TODO add all the default parameters here
pipeline_parameters = {
    'clf__max_depth': 8,
    'clf__n_estimators': 10,
}


def get_best_parameters_grid(parameter_grid, articles, categories):
    # find the best parameters for both the feature extraction and the
    # increase processing time in a combinatorial way
    grid_search = GridSearchCV(pipeline, parameter_grid, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("Pipeline:", [name for name, _ in pipeline.steps])
    print("Parameter grid:")
    pprint(parameter_grid)
    t0 = datetime.now()
    grid_search.fit(articles, categories)
    print("done in {}".format(datetime.now() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameter_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return best_parameters


def load_data(file):
    log.debug("Loading data...")

    # gives TextFileReader, which is iterable with chunks of 1000 rows.
    tp = pd.read_csv(file, iterator=True, chunksize=1000)
    # df is DataFrame. If errors, do `list(tp)` instead of `tp`
    df = pd.concat(tp, ignore_index=True)

    return df.loc[:, ['Id', 'Title', 'Abstract']], df.loc[:, 'Category'] if 'Category' in df else None


def filtered(params):
    """Returns a new dictionary with union and steps omitted."""
    filtered_params = { key: value for key, value in params.items()
                        if not isinstance(value, Pipeline)
                        and not isinstance(value, FeatureUnion)
                        and not isinstance(value, HashingVectorizer)
                        and not isinstance(value, RandomForestClassifier)
                        and not key.endswith('steps')
                        and not key.endswith('transformer_list')}
    return filtered_params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'), help='The training data to be used to create a model. The created model <timestamp>.model is savede to disk.')
    parser.add_argument('--grid', action='store_true', help='Whether or not to use grid search to get the optimal hyper-parameter configuration. See http://scikit-learn.org/stable/modules/grid_search.html#grid-search')
    parser.add_argument('--model', metavar='FILE', type=argparse.FileType('r'), help='The model to be used for classification.')
    parser.add_argument('--predict', metavar='FILE', type=argparse.FileType('r'), help='Data to be classified.')
    #parser.print_help()

    args = parser.parse_args()

    print(args)

    if (args.train and args.model) \
    or (not args.train and not args.model):
        print("Invalid mode of operation!")
        parser.print_help()
        exit(1)

    pipeline = joblib.load(args.model) if args.model else Pipeline([
        # Use FeatureUnion to combine the features
        ('union', FeatureUnion([

                # Pipeline for pulling features from the articles's title
                ('title', Pipeline([
                    ('selector', ItemSelector(key='Title')),        # ('printer', Printer()),
                    ('hasher', HashingVectorizer()),                # ('printer', Printer()),
                ])),
                ('abstract', Pipeline([
                    ('selector', ItemSelector(key='Abstract')),     # ('printer', Printer()),
                    ('hasher', HashingVectorizer()),                # ('printer', Printer()),
                ])),
                #TODO add your feature vectors here
                # Pipeline for pulling features from the articles's abstract
                #('title', Pipeline([
                #    ('selector', ItemSelector(key='abstract')),
                #    ('hasher', HashingVectorizer()),
                #])),
            ],
        )),
        #('printer', Printer()),
        ('clf', RandomForestClassifier()),
    ])

    if args.train:
        log.info("Fitting...")

        articles, categories = load_data(args.train)

        # Find the best hyper-parameter configuration or use the defined one.
        if args.grid:
            log.info("Using grid search to find the best hyper parameter configuration...")
            best_pipeline_parameters = get_best_parameters_grid(pipeline_parameters_grid, articles, categories)
            pipeline.set_params(**best_pipeline_parameters)
        else:
            log.info("Using the defined hyper parameter configuration...")
            pipeline.set_params(**pipeline_parameters)

        pipeline.fit(articles, categories)

        y_true, y_pred = categories, pipeline.predict(articles)

        print()
        print("Crossvalidation report:")
        print()
        print("Hyper parameters:")
        pprint(filtered(pipeline.get_params()))
        print()
        print(classification_report(y_true, y_pred))

        model_filename = 'model_{}.pkl'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
        joblib.dump(pipeline, model_filename)

    if args.predict:
        log.info("Predicting...")

        articles, _ = load_data(args.predict)

        if args.train:
            log.info("Using hyper parameter as of the fitting phase...")
            # The hyper-parameter configuration was already set.
        else:
            log.info("Using the defined hyper parameter configuration...")
            pipeline.set_params(**pipeline_parameters)

        categories = pipeline.predict(articles)

        print("Classification report:")
        print()
        print("Hyper parameters:")
        pprint(filtered(pipeline.get_params()))
        print()
        print(articles.assign(Category=pd.Series(categories).values))
