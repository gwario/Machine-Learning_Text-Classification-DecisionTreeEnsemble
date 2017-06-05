#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import logging as log
from sys import exit
import argparse
from datetime import datetime
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import config as cfg
import report as rp
import file_io as io

__doc__ = """
Classifies, train and saves the model.

Modes of operation:

Score, train and predict:   (--score --train --predict)
    Evaluates the classifier, trains it (model saved to disk) and predicts (prediction saved to disk).
    
Score and train:            (--score --train)
    Evaluates the classifier and trains it (model saved to disk).
    
Train and predict:          (--train --predict)
    Trains the classifier (model saved to disk) and predicts (prediction saved to disk).
    
Load model and predict:     (--model --predict)
    Loads the model and predicts (prediction saved to disk).
    
All modes of operation support model selection with --hp.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"

# Display progress logs on stdout
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


# For passing in floating point ranges as cl arguments
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


def get_grid_result(pipeline, parameter_grid, hp_metric, x, y):
    """ Finds and returns the best parameters for both the feature extraction and the classification.
    Changing the grid increases processing time in a combinatorial way."""

    log.debug("Performing grid search, optimizing {} score...".format(hp_metric))
    grid_search = GridSearchCV(pipeline, parameter_grid, scoring=hp_metric, n_jobs=-1, verbose=1)

    t0 = datetime.now()
    grid_search.fit(x, y)
    dt_grid = datetime.now() - t0

    best_parameters = grid_search.best_estimator_.get_params()

    rp.print_hyper_parameter_search_report_grid(pipeline, dt_grid, parameter_grid, grid_search.best_score_, best_parameters)

    return best_parameters

def get_randomized_result(pipeline, parameter_randomized, hp_metric, x, y):
    
    log.debug("Performing randomized search, optimizing {} score...".format(hp_metric))
    randomized_search = RandomizedSearchCV(pipeline, parameter_randomized, scoring=hp_metric, 
            n_iter = n_iter_search, n_jobs=-1, verbose=1)

    t0 = datetime.now()
    randomized_search.fit(x, y)
    dt_randomized = datetime.now() - t0

    best_parameters = randomized_search.best_estimator_.get_params()
    rp.print_hyper_parameter_search_report_randomized(pipeline, dt_randomized, parameter_randomized, randomized_search.best_score_, best_parameters)

    return best_parameters


def get_args(args_parser):
    """Parses and returns the command line arguments."""

    args_parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'),
                             help='''The training data to be used to create a model. The created model <timestamp>.model
                              is saved to disk.''')
    args_parser.add_argument('--hp', metavar='METHOD', nargs='?', const='config', default='config',
                             choices=['config', 'grid'],
                             help='''The method to get the hyper-parameters. One of 'config' (use the pre-defined
                             configuration in config.py) or 'grid' (GridSearchCV). (default: '%(default)s' ''')
    args_parser.add_argument('--hp_metric', metavar='METRIC', nargs='?', const='f1_macro', default='f1_macro',
                             choices=['accuracy', 'average_precision', 'f1', 'precision', 'recall',
                                      'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
                                      'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
                                      'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
                                      'roc_auc'],
                             help='''The metric to use for the hyper-parameter optimization. Used with 'grid'.
                             (default: '%(default)s' ''')
    args_parser.add_argument('--score', action='store_true',
                             help='''Whether or not to evaluate the estimator performance.''')
    args_parser.add_argument('--test_size', metavar='FRACTION', type=float, choices=[Range(0.0, 1.0)], default=0.25,
                             help='Size of the test/train split, as a fraction of the total data.')
    args_parser.add_argument('--model', metavar='FILE', type=argparse.FileType('r'),
                             help='The model to be used for classification.')
    args_parser.add_argument('--predict', metavar='FILE', type=argparse.FileType('r'), help='Data to be classified.')
    # parser.print_help()

    return args_parser.parse_args()


def check_mode_of_operation(arguments):
    """Checks the mode of operation."""

    if (arguments.score and arguments.train and arguments.predict) \
            or (arguments.score and arguments.train) \
            or (arguments.train and arguments.predict) \
            or (arguments.model and arguments.predict):
        log.debug("Valid mode of operation")
    else:
        print("Invalid mode of operation!")
        parser.print_help()
        exit(1)


def get_configuration(data_set):
    """Returns the right pipeline and parameters for the given data-set."""

    if data_set == 'binary':
        return cfg.binary_pipeline, cfg.binary_pipeline_parameters, cfg.binary_pipeline_parameters_grid, cfg.binary_pipeline_parameters_randomized

    elif data_set == 'multi-class':
        return cfg.multiclass_pipeline, cfg.multiclass_pipeline_parameters, cfg.multiclass_pipeline_parameters_grid, cfg.multiclass_pipeline_parameters_randomized

def get_optimized_parameters_grid(pipeline, x_train, y_train, hp_metric, pipeline_parameters_grid):

    # Find the best hyper-parameter configuration or use the defined one.
    if args.hp == 'grid':
        log.info("Using grid search on the training set to select the best model (hyper-parameters)...")

        return get_grid_result(pipeline, pipeline_parameters_grid, hp_metric, x_train, y_train)

def get_optimized_parameters_randomized(pipeline, x_train, y_train, hp_metric, pipeline_parameters_randomized):
    
    # Find the best hyper-parameter configuration or use the defined one.
    if args.hp == 'randomized':
        log.info("Using randomized search on the training set to select the best model (hyper-parameters)...")

        return get_randomized_result(pipeline, pipeline_parameters_randomized, hp_metric, x_train, y_train)


def mode_score(pipeline, x_train, y_train, x_test, y_test):
    """Evaluates the estimator.

    Returns
    -------
    pipeline: The parametrized, unfitted pipeline.
    """

    log.info("Evaluating the selected model on the test set...")
    # The pipeline parameters
    rp.print_hyper_parameters(pipeline)

    tFit = datetime.now()
    pipeline.fit(x_train, y_train)
    rp.print_fitting_report(pipeline,
                            dt_fitting=datetime.now() - tFit,
                            x_train=x_train,
                            y_train=y_train)

    tPredict = datetime.now()
    categories_true, categories_predicted = y_test, pipeline.predict(x_test)
    rp.print_evaluation_report(pipeline,
                               dt_evaluation=datetime.now() - tPredict,
                               y_true=categories_true,
                               y_pred=categories_predicted)

    # Reset the estimator to the state be for fitting
    pipeline = clone(pipeline)

    return pipeline


def select_model(args, data_set, x_train, y_train):
    """Selects a model according to args.hp and args.hp_metric either by using the predefined ones or by using a
    cross-validated search strategy.
    The model is selected based on the args.train data-set.

    Returns
    -------
    pipeline: The parametrized pipeline.
    """

    pipeline, pipeline_parameters, pipeline_parameters_grid, pipeline_parameters_randomized = get_configuration(data_set)

    if args.hp == 'config':
        log.info("Using the pre-selected model (hyper-parameters)...")
        pipeline.set_params(**pipeline_parameters)

    elif args.hp == 'grid':
        best_params = get_optimized_parameters_grid(pipeline, x_train, y_train, args.hp_metric, pipeline_parameters_grid)
        pipeline.set_params(**best_params)
        
        # Reset the estimator to the state be for fitting
        pipeline = clone(pipeline)

    elif args.hp == 'randomized':
        best_params = get_optimized_parameters_randomized(pipeline, x_train, y_train, args.hp_metric, pipeline_parameters_randomized)
        pipeline.set_params(**best_params)
        pipeline = clone(pipeline)

    return pipeline


def mode_train(pipeline, x, y):
    """Trains the model.

    Returns
    -------
    pipeline: The parametrized and fitted pipeline.
    """
    model_filename = 'model_{}.pkl'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

    log.info("Building the selected model on the whole data set...")
    # The pipeline parameters
    rp.print_hyper_parameters(pipeline)

    t_fit = datetime.now()
    pipeline.fit(x, y)
    rp.print_fitting_report(pipeline,
                            dt_fitting=datetime.now() - t_fit,
                            x_train=x,
                            y_train=y)

    io.save_model(pipeline, model_filename)

    return pipeline


def mode_predict(pipeline, x):
    """Predicts using the given model."""

    prediction_filename = 'prediction_{}.csv'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

    log.info("Predicting {} data points...".format(len(x)))

    t_predict = datetime.now()
    y = pipeline.predict(x)
    combined_data = x.assign(Category=pd.Series(y).values)
    rp.print_prediction_report(pipeline,
                               dt_predict=datetime.now() - t_predict,
                               data=combined_data)

    io.save_prediction(combined_data.loc[:, ['Id', 'Category']], prediction_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = get_args(parser)
    log.debug("Commandline arguments: {}".format(args))

    check_mode_of_operation(args)

    if args.score and args.train and args.predict:
        log.info("Mode score->train->predict")

        x, y = io.load_data(args.train)

        # Create a training/validation and a test set for model selection (hyper-parameter search) and evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=args.test_size,
                                                            random_state=cfg.split_random_state)
                                                            #stratify=y)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.predict)
        pipeline = select_model(args, data_set, x_train, y_train)

        # Score part
        pipeline = mode_score(pipeline, x_train, y_train, x_test, y_test)

        # Train part
        mode_train(pipeline, x, y)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(pipeline, x)

    elif args.score and args.train:
        log.info("Mode score->train")

        x, y = io.load_data(args.train)

        # Create a training/validation and a test set for model selection (hyper-parameter search) and evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=args.test_size,
                                                            random_state=cfg.split_random_state)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.train)
        pipeline = select_model(args, data_set, x_train, y_train)

        # Score part
        pipeline = mode_score(pipeline, x_train, y_train, x_test, y_test)

        # Train part
        mode_train(pipeline, x, y)

    elif args.train and args.predict:
        log.info("Mode train->predict")

        x, y = io.load_data(args.train)

        # Create a training/validation and a test set for model selection (hyper-parameter search) and evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=args.test_size,
                                                            random_state=cfg.split_random_state)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.predict)
        pipeline = select_model(args, data_set, x_train, y_train)

        # Train part
        mode_train(pipeline, x, y)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(pipeline, x)

    elif args.model and args.predict:
        log.info("Mode model->predict")

        # Load model
        pipeline = io.load_model(args.model)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(pipeline, x)
