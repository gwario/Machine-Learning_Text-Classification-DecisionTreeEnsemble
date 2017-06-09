import logging as log
import re
from pprint import pprint
from pprint import pformat
from pandas import DataFrame

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

from preprocessor import NLTKPreprocessor, WordNetLemmatizer

__doc__ = """
Contains code to print results of fitting, predicting or cross-validation phases.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


def filtered_search(parameter_search):
    return parameter_search


def filtered(params):
    """Returns a new parameter dictionary with non-parameters omitted."""

    filtered_params = {key: value for key, value in params.items()
                       if not isinstance(value, Pipeline)
                       and not isinstance(value, FeatureUnion)
                       and not isinstance(value, HashingVectorizer)
                       and not isinstance(value, CountVectorizer)
                       and not isinstance(value, RandomForestClassifier)
                       and not isinstance(value, NLTKPreprocessor)
                       and not isinstance(value, WordNetLemmatizer)
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


def reduce_object(string, object, ellipsis="(...)"):
    """Replaces there object parameters with ellipsis."""

    matches = re.compile(object).finditer(string)

    for match in reversed(list(matches)):
        i = match.end()
        cnt = 0
        while cnt != 0 or i == match.end():
            assert i < len(string)

            if string[i] == '(':
                cnt += 1
            elif string[i] == ')':
                cnt -= 1
            i += 1
        # i is pos after closing bracket
        string = string[:match.end()] + ellipsis + string[i:]

    return string


def reduce_objects(string, objects, ellipsis="(...)"):

    for object in objects:
        string = reduce_object(string, object, ellipsis)

    return string


def print_search_space(parameter_search_space):
    """Prints the parameter search space."""

    string = pformat(parameter_search_space)
    string = reduce_objects(string, ['NLTKPreprocessor', 'CountVectorizer'])

    print("Parameter search space:")
    print(string)


def print_best_parameters(search, search_space):
    """Prints the best parameters."""

    print("Best parameters set:")
    for param_name in sorted(search_space.keys()):
        param_value = pformat(search.best_params_[param_name])
        param_value = reduce_objects(param_value, ['NLTKPreprocessor', 'CountVectorizer'])
        print("\t%s: %s" % (param_name, param_value))


def print_folds_results(search):
    """Prints the configuration and statistics of each fold."""

    print("Detailed folds results:")
    print(DataFrame(search.cv_results_))


def print_hyper_parameter_search_report_grid(pipeline, dt_search, parameter_grid, grid_search):
    """Prints the i.e. grid search report."""

    print("Hyper parameter search report:")
    print_search_space(parameter_grid)

    print("Best score: %0.3f" % grid_search.best_score_)

    print_best_parameters(grid_search, parameter_grid)
    print_folds_results(grid_search)

    log.debug("Grid search done in {}".format(dt_search))


def print_hyper_parameter_search_report_randomized(pipeline, dt_search, parameter_randomized, randomized_search):
    """Prints the i.e. randomized search report."""

    print("Hyper parameter search report:")
    print_search_space(parameter_randomized)

    print("Best score: %0.3f" % randomized_search.best_score_)

    print_best_parameters(randomized_search, parameter_randomized)
    print_folds_results(randomized_search)

    log.debug("Randomized search done in {}".format(dt_search))


def print_hyper_parameters(pipeline):
    """Prints the i.e. grid search report."""

    print("Hyper parameters:")
    pprint(filtered(pipeline.get_params()))
