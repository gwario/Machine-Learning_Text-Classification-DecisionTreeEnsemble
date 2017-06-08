from datetime import datetime
import logging as log

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import config as cfg
import report as rp

__doc__ = """
Contains code to handle hyper-parameter optimization.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


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
                                           random_state=cfg.pipeline_parameters_randomized_random_state,
                                           n_iter=cfg.pipeline_parameters_randomized_n_iter,
                                           refit=False,
                                           n_jobs=-1, verbose=1)

    t0 = datetime.now()
    randomized_search.fit(x, y)
    dt_randomized = datetime.now() - t0

    best_parameters = randomized_search.best_estimator_.get_params()
    rp.print_hyper_parameter_search_report_randomized(pipeline, dt_randomized, parameter_randomized, randomized_search.best_score_, best_parameters)

    return best_parameters


def get_optimized_parameters_grid(pipeline, x_train, y_train, hp_metric, pipeline_parameters_grid):

    log.info("Using grid search on the training set to select the best model (hyper-parameters)...")

    return get_grid_result(pipeline, pipeline_parameters_grid, hp_metric, x_train, y_train)


def get_optimized_parameters_randomized(pipeline, x_train, y_train, hp_metric, pipeline_parameters_randomized):

    log.info("Using randomized search on the training set to select the best model (hyper-parameters)...")

    return get_randomized_result(pipeline, pipeline_parameters_randomized, hp_metric, x_train, y_train)
