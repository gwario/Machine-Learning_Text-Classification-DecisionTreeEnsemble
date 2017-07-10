#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function, absolute_import

import argparse
import logging as log
from datetime import datetime
from sys import exit

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import config as cfg
import file_io as io
import hyper_parameter_search as hp
import report as rp
from config import PipelineConfiguration

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


def get_args(args_parser):
    """Parses and returns the command line arguments."""

    args_parser.add_argument('--train', metavar='FILE', type=argparse.FileType('r'),
                             help='''The training data to be used to create a model. The created model <timestamp>.model
                              is saved to disk.''')
    args_parser.add_argument('--hp', metavar='METHOD', nargs='?', const='config', default='config',
                             choices=['config', 'grid', 'randomized', 'evolutionary'],
                             help='''The method to get the hyper-parameters. One of 'config' (use the pre-defined
                             configuration in config.py), 'evolutionary' (EvolutionaryAlgorithmSearchCV), 'randomized' (RandomizedSearchCV) or 'grid' (GridSearchCV).
                             (default: '%(default)s' ''')
    args_parser.add_argument('--hp_metric', metavar='METRIC', nargs='?', const='accuracy', default='accuracy',
                             choices=['accuracy', 'average_precision', 'f1', 'precision', 'recall',
                                      'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
                                      'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
                                      'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
                                      'roc_auc'],
                             help='''The metric to use for the hyper-parameter optimization. Used with 'grid',
                             'randomized' and 'evolutionary'. (default: '%(default)s' 
                             F1 = 2 * (precision * recall) / (precision + recall)
                             In the multi-class and multi-label case, this is the weighted average of the F1 score of
                             each class.''')
    args_parser.add_argument('--oob', action='store_true',
                             help='''Whether or not to Calculate the out of bag score for the estimator.''')
    
    args_parser.add_argument('--importance', action='store_true',
                             help='''Whether or not to Calculate the importance of features.''')

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


def mode_score(args, fu_pl, clf_pl, x_train, y_train, x_test, y_test):
    """Evaluates the estimator.

    Returns
    -------
    pipeline: The parametrized, unfitted pipeline.
    """

    log.info("Evaluating the selected model on the test set...")

    tFit = datetime.now()

    log.debug("Generating feature vector...")
    t0 = datetime.now()
    x_train = fu_pl.fit_transform(x_train)
    log.info("Generated vector of {} features in {} from {} samples.".format(x_train.shape[1], datetime.now() - t0, x_train.shape[0]))

    if args.oob:
        clf_n_min = 25
        clf_n_max = clf_pl.get_params()['clf__n_estimators']
        oob_scores = []

        bs = clf_pl.get_params()['clf__bootstrap']
        clf_pl.set_params(clf__bootstrap=True)
        clf_pl.set_params(clf__oob_score=True)
        clf_pl.set_params(clf__warm_start=True)

        log.debug("Calculating out-of-bag score over tree count (range {}-{})...".format(clf_n_min, clf_n_max))
        for i in range(clf_n_min, clf_n_max + 1):
            clf_pl.set_params(clf__n_estimators=i)
            clf_pl.fit(x_train, y_train)
            oob_scores.append((i, clf_pl.get_params()['clf'].oob_score_))
            if i % 10 == 0:
                log.debug("Out-of-bag score for {} trees = {}".format(i, clf_pl.get_params()['clf'].oob_score_))

        io.store_oob_score_data(clf_pl.get_params(), oob_scores)

        clf_pl.set_params(clf__bootstrap=bs)
        clf_pl.set_params(clf__oob_score=False)
        clf_pl.set_params(clf__warm_start=False)
        clf_pl.set_params(clf__n_estimators=clf_n_max)

    else:
        clf_n_min = None
        clf_n_max = None
        oob_scores = None
        clf_pl.fit(x_train, y_train)

    rp.print_fitting_report(clf_pl,
                            dt_fitting=datetime.now() - tFit,
                            x_train=x_train,
                            y_train=y_train,
                            min_estimators=clf_n_min,
                            max_estimators=clf_n_max,
                            score=oob_scores)
    
    # --- Feature Importances ---
    if args.importance:
        clf_n = clf_pl.get_params()['clf__n_estimators']

        log.debug("Compute the feature importances...")
        clf_pl.set_params(clf__n_estimators=clf_n)
        clf_pl.fit(x_train, y_train)

        importances = clf_pl.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf_pl.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        rp.print_feature_importances_report(clf_pl,
                                            dt_fitting=datetime.now() - tFit,
                                            x_train=x_train,
                                            y_train=y_train,
                                            n_estimators=clf_n,
                                            p_importances=importances,
                                            p_indices=indices)

    else:
        clf_pl.fit(x_train, y_train)

        rp.print_feature_importances_report(clf_pl,
                                            dt_fitting=datetime.now() - tFit,
                                            x_train=x_train,
                                            y_train=y_train)

        

    #--- End ---

    log.debug("Generating feature vector...")
    t0 = datetime.now()
    x_test = fu_pl.transform(x_test)
    log.info("Generated vector of {} features in {} from {} samples.".format(x.shape[1], datetime.now() - t0, x.shape[0]))

    tPredict = datetime.now()
    categories_true, categories_predicted = y_test, clf_pl.predict(x_test)
    rp.print_evaluation_report(clf_pl,
                               dt_evaluation=datetime.now() - tPredict,
                               y_true=categories_true,
                               y_pred=categories_predicted)

    # Reset the estimator to the state be for fitting
    clf_pl = clone(clf_pl)

    return fu_pl, clf_pl


def select_model(args, data_set, x_train, y_train):
    """Selects a model according to args.hp and args.hp_metric either by using the predefined ones or by using a
    cross-validated search strategy.
    The model is selected based on the args.train data-set.

    Returns
    -------
    pipeline: The parametrized pipeline.
    """
    configuration = PipelineConfiguration(data_set)
    (fu_pl, clf_pl) = configuration.pipelines()

    if args.hp == 'config':
        log.info("Using the pre-selected model (hyper-parameters)...")
        clf_pl.set_params(**configuration.parameters())

    elif args.hp == 'grid' or args.hp == 'randomized' or args.hp == 'evolutionary':
        best_params = hp.get_optimized_parameters(args.hp, configuration, x_train, y_train, args.hp_metric)
        clf_pl.set_params(**best_params)
        
        # Reset the estimator to the state be for fitting
        clf_pl = clone(clf_pl)

    return fu_pl, clf_pl


def mode_train(fu_pl, clf_pl, x, y):
    """Trains the model.

    Returns
    -------
    pipeline: The parametrized and fitted pipeline.
    """
    model_filename = 'model_{}.pkl'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

    log.info("Building the selected model on the whole data set...")
    # The pipeline parameters
    rp.print_hyper_parameters(clf_pl)

    log.debug("Generating feature vector...")
    t0 = datetime.now()
    x = fu_pl.fit_transform(x)
    log.info("Generated vector of {} features in {} from {} samples.".format(x.shape[1], datetime.now() - t0, x.shape[0]))

    t_fit = datetime.now()
    clf_pl.fit(x, y)
    rp.print_fitting_report(clf_pl,
                            dt_fitting=datetime.now() - t_fit,
                            x_train=x,
                            y_train=y)

    io.save_model(fu_pl, clf_pl, model_filename)

    return fu_pl, clf_pl


def mode_predict(fu_pl, clf_pl, x):
    """Predicts using the given model."""

    prediction_filename = 'prediction_{}.csv'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

    log.info("Predicting {} data points...".format(len(x)))

    log.debug("Generating feature vector...")
    t0 = datetime.now()
    x_feature_vect = fu_pl.transform(x)
    log.info("Generated vector of {} features in {} from {} samples.".format(x_feature_vect.shape[1], datetime.now() - t0, x_feature_vect.shape[0]))

    t_predict = datetime.now()
    y = clf_pl.predict(x_feature_vect)
    combined_data = x.assign(Category=pd.Series(y).values)
    rp.print_prediction_report(clf_pl,
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
                                                            random_state=cfg.split_random_state,
                                                            stratify=y)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.predict)
        fu_pl, clf_pl = select_model(args, data_set, x_train, y_train)

        # Score part
        fu_pl, clf_pl = mode_score(args, fu_pl, clf_pl, x_train, y_train, x_test, y_test)

        # Train part
        mode_train(fu_pl, clf_pl, x, y)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(fu_pl, clf_pl, x)

    elif args.score and args.train:
        log.info("Mode score->train")

        x, y = io.load_data(args.train)

        # Create a training/validation and a test set for model selection (hyper-parameter search) and evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=args.test_size,
                                                            random_state=cfg.split_random_state,
                                                            stratify=y)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.train)
        fu_pl, clf_pl = select_model(args, data_set, x_train, y_train)

        # Score part
        fu_pl, clf_pl = mode_score(args, fu_pl, clf_pl, x_train, y_train, x_test, y_test)

        # Train part
        mode_train(fu_pl, clf_pl, x, y)

    elif args.train and args.predict:
        log.info("Mode train->predict")

        x, y = io.load_data(args.train)

        # Create a training/validation and a test set for model selection (hyper-parameter search) and evaluation
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=args.test_size,
                                                            random_state=cfg.split_random_state,
                                                            stratify=y)

        log.info("Created training set ({}) and test set ({})".format(len(y_train), len(y_test)))

        data_set = io.get_data_set(args.predict)
        fu_pl, clf_pl = select_model(args, data_set, x_train, y_train)

        # Train part
        mode_train(fu_pl, clf_pl, x, y)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(fu_pl, clf_pl, x)

    elif args.model and args.predict:
        log.info("Mode model->predict")

        # Load model
        fu_pl, clf_pl = io.load_model(args.model)

        # Predict part
        x, _ = io.load_data(args.predict)

        mode_predict(fu_pl, clf_pl, x)
