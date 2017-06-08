import logging as log
from pprint import pprint

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

"""
Contains code to print results of fitting, predicting or cross-validation phases.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


def filtered(params):
    """Returns a new parameter dictionary with non-parameters omitted."""

    filtered_params = {key: value for key, value in params.items()
                       if not isinstance(value, Pipeline)
                       and not isinstance(value, FeatureUnion)
                       and not isinstance(value, HashingVectorizer)
                       and not isinstance(value, CountVectorizer)
                       and not isinstance(value, RandomForestClassifier)
                       and not key.endswith('steps')
                       and not key.endswith('transformer_list')}

    return filtered_params


def print_fitting_report(pipeline, dt_fitting, x_train, y_train):
    """Prints the training report."""

    log.debug("Fitted {} data points.".format(len(y_train)))
    log.debug("Fitting done in {}".format(dt_fitting))


def print_evaluation_report(pipeline, dt_evaluation, y_pred, y_true):
    """Prints the cross-validation report."""

    print("Evaluation report:")
    print(classification_report(y_true, y_pred))
    log.debug("Evaluation done in {}".format(dt_evaluation))

    # TODO Improve visualization of the validation report i.e. add useful metrics from http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics


def print_prediction_report(pipeline, dt_predict, data):
    """Prints the classification result."""

    # TODO Improve visualization of the result report i.e. statistics on how many documents per class...

    #print("Classification report:")
    # print(data)
    log.debug("Prediction done in {}".format(dt_predict))


def print_hyper_parameter_search_report_grid(pipeline, dt_search, parameter_grid, best_score, best_parameters):
    """Prints the i.e. grid search report."""

    print("Hyper parameter search report:")
    print("Parameter grid: {}".format(parameter_grid))
    print("Best score: %0.3f" % best_score)
    print("Best parameters set:")
    for param_name in sorted(parameter_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    log.debug("Grid search done in {}".format(dt_search))


def print_hyper_parameter_search_report_randomized(pipeline, dt_search, parameter_randomized, best_score, best_parameters):
    """Prints the i.e. randomized search report."""

    print("Hyper parameter search report:")
    print("Parameter grid: {}".format(parameter_randomized))
    print("Best score: %0.3f" % best_score)
    print("Best parameters set:")
    for param_name in sorted(parameter_randomized.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    log.debug("Randomized search done in {}".format(dt_search))


def print_hyper_parameters(pipeline):
    """Prints the i.e. grid search report."""

    print("Hyper parameters:")
    pprint(filtered(pipeline.get_params()))
