#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import logging as log
from sys import exit
import argparse
from datetime import datetime
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import config as cfg
import report as rp
import file_io as io

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
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"

# Display progress logs on stdout
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def get_best_parameters_grid(parameter_grid, articles, categories):
    """ Finds and returns the best parameters for both the feature extraction and the classification.
    Changing the grid increases processing time in a combinatorial way."""

    log.debug("Performing grid search...")
    grid_search = GridSearchCV(pipeline, parameter_grid, n_jobs=-1, verbose=1)

    t0 = datetime.now()
    grid_search.fit(articles, categories)
    dt_grid = datetime.now() - t0

    best_parameters = grid_search.best_estimator_.get_params()

    rp.print_hyper_parameter_search_report(pipeline, dt_grid, parameter_grid, grid_search.best_score_, best_parameters)

    return best_parameters


def get_args(parser):
    """Parses and returns the command line arguments."""

    parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'),
                        help='''The training data to be used to create a model. The created model <timestamp>.model is 
                        saved to disk.''')
    parser.add_argument('--grid', action='store_true',
                        help='''Whether or not to use grid search to get the optimal hyper-parameter configuration. 
                        See http://scikit-learn.org/stable/modules/grid_search.html#grid-search''')
    parser.add_argument('--model', metavar='FILE', type=argparse.FileType('r'),
                        help='The model to be used for classification.')
    parser.add_argument('--predict', metavar='FILE', type=argparse.FileType('r'), help='Data to be classified.')
    # parser.print_help()

    return parser.parse_args()


def get_data_set(data_set):
    """Returns the data-set identification string. One of 'binary', 'multi-class' or, in case of an invalid datas-et,
    'unknown_data_set'"""

    first_line = next(data_set)
    data_set.seek(0)

    if 'Title' in first_line and 'Abstract' in first_line:
        return 'binary'
    elif 'Text' in first_line:
        return 'multi-class'
    else:
        return 'unknown_data_set'


def check_mode_of_operation(args):
    """Checks the mode of operation."""

    if (args.train and args.model) \
            or (args.model and not args.predict) \
            or (args.predict and not args.train and not args.model):

        print("Invalid mode of operation!")
        parser.print_help()
        exit(1)


def get_configuration(args):
    """Checks the supplied data-sets and returns the right pipeline and parameters."""

    data_set_train = ''
    data_set_predict = ''

    if args.train:
        data_set_train = get_data_set(args.train)
        if data_set_train not in ['binary', 'multi-class']:
            print("Training data-set invalid!")
            exit(1)

    if args.predict:
        data_set_predict = get_data_set(args.predict)
        if data_set_predict not in ['binary', 'multi-class']:
            print("Prediction data-set invalid!")
            exit(1)

    if args.train and args.predict and data_set_train is not data_set_predict:
        print("Training data-set does not match prediction data-set!")
        exit(1)

    if data_set_train is 'binary' or data_set_predict is 'binary':

        log.info("Recognised data-set: binary")
        return io.load_model(args.model) if args.model else cfg.binary_pipeline, \
            cfg.binary_pipeline_parameters, cfg.binary_pipeline_parameters_grid

    else:

        log.info("Recognised data-set: multi-class")
        return io.load_model(args.model) if args.model else cfg.multiclass_pipeline, \
            cfg.multiclass_pipeline_parameters, cfg.multiclass_pipeline_parameters_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = get_args(parser)
    log.debug("Commandline arguments: {}".format(args))

    check_mode_of_operation(args)

    pipeline, pipeline_parameters, pipeline_parameters_grid = get_configuration(args)

    if args.train:
        log.info("Fitting...")

        articles, categories = io.load_data(args.train)

        # Find the best hyper-parameter configuration or use the defined one.
        if args.grid:
            log.info("Using grid search to find the best hyper parameter configuration...")
            best_pipeline_parameters = get_best_parameters_grid(pipeline_parameters_grid, articles, categories)
            pipeline.set_params(**best_pipeline_parameters)
        else:
            log.info("Using the defined hyper parameter configuration...")
            pipeline.set_params(**pipeline_parameters)

        # TODO Do real cross-validation here: http://scikit-learn.org/stable/modules/cross_validation.html

        t0 = datetime.now()
        pipeline.fit(articles, categories)
        dtFit = datetime.now() - t0

        t0 = datetime.now()
        cv_scores = cross_val_score(pipeline, articles, categories, cv=5)
        dtCV = datetime.now() - t0

        t0 = datetime.now()
        categories_true, categories_predicted = categories, pipeline.predict(articles)
        dtValid = datetime.now() - t0

        filename = 'model_{}.pkl'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
        io.save_model(pipeline, filename)

        rp.print_training_report(pipeline, cv_scores, dtFit, dtValid, articles, categories_true, categories_predicted)

    if args.predict:
        log.info("Predicting...")

        articles, _ = io.load_data(args.predict)

        if args.train:
            log.info("Using hyper parameter as of the fitting phase...")
            # The hyper-parameter configuration was already set.
        else:
            log.info("Using the defined hyper parameter configuration...")
            pipeline.set_params(**pipeline_parameters)

        t0 = datetime.now()
        categories = pipeline.predict(articles)
        dtPredict = datetime.now() - t0

        prediction = articles.assign(Category=pd.Series(categories).values)

        rp.print_prediction_report(pipeline, dtPredict, prediction)

        filename = 'prediction_{}.csv'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
        io.save_prediction(prediction.loc[:, ['Id', 'Category']], filename)

