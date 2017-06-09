import logging as log
from numpy.ma import MaskedArray
from copy import deepcopy
from pprint import pprint
from pandas import DataFrame

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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


def filtered_best_params(params):
    """Returns a new parameter dictionary with Pipeline values replaced by 'Pipeline'."""

    filtered_params = {key: value for key, value in params.items()
                       if not isinstance(value, FeatureUnion)
                       and not isinstance(value, HashingVectorizer)
                       and not isinstance(value, CountVectorizer)
                       and not isinstance(value, RandomForestClassifier)
                       and not isinstance(value, NLTKPreprocessor)
                       and not isinstance(value, WordNetLemmatizer)
                       and not key.endswith('steps')
                       and not key.endswith('transformer_list')}

    filtered_params = {key: (value if not isinstance(value, Pipeline) else 'Pipeline')
                       for key, value in filtered_params.items()}

    return filtered_params


def filtered_search_space(param_search_space):
    """Returns a new parameter dictionary with Pipeline values replaced by 'Pipeline'."""

    tuple_pipeline_string = lambda tpl: tuple(('Pipeline'
                                             if isinstance(value, Pipeline)
                                             else value
                                             for value in tpl))

    contains_pipeline = lambda value: isinstance(value, tuple) and any(isinstance(val, Pipeline) for val in value)

    filtered_params = {key: (tuple_pipeline_string(value)
                             if contains_pipeline(value)
                             else value)
                       for key, value in param_search_space.items()}

    return filtered_params


def print_fitting_report(pipeline, dt_fitting, x_train, y_train):
    """Prints the training report."""

    log.debug("Fitted {} data points.".format(len(y_train)))
    log.debug("Fitting done in {}".format(dt_fitting))


def print_evaluation_report(pipeline, dt_evaluation, y_pred, y_true):
    """Prints the cross-validation report."""

    print("Evaluation report:")
    print(classification_report(y_true, y_pred))
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    log.debug("Evaluation done in {}".format(dt_evaluation))


def print_prediction_report(pipeline, dt_predict, data):
    """Prints the classification result."""

    # TODO Improve visualization of the result report i.e. statistics on how many documents per class...

    #print("Classification report:")
    # print(data)
    log.debug("Prediction done in {}".format(dt_predict))


def print_search_space(parameter_search_space):
    """Prints the parameter search space."""

    print("Parameter search space:")
    pprint(filtered_search_space(parameter_search_space))


def print_best_parameters(search, search_space):
    """Prints the best parameters."""

    print("Best parameters set:")
    pprint(filtered_best_params(search.best_params_))


def print_folds_results(search):
    """Prints the configuration and statistics of each fold."""

    cv_results = deepcopy(search.cv_results_)

    print("Detailed folds results:")
    # Remove the redundant params list
    cv_results.pop('params', None)
    # Replace pipelines
    for param_key in cv_results:
        if isinstance(cv_results[param_key], MaskedArray):
            for index, value in enumerate(cv_results[param_key].data):
                if isinstance(value, Pipeline):
                    cv_results[param_key].data[index] = 'Pipeline'

    print(DataFrame(cv_results))


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
