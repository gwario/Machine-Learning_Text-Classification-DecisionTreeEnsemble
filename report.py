import logging as log
from numpy.ma import MaskedArray
from copy import deepcopy
from pprint import pprint
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

import file_io as io
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
                       and not isinstance(value, SelectKBest)
                       and not isinstance(value, ExtraTreesClassifier)
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
                       and not isinstance(value, ExtraTreesClassifier)
                       and not isinstance(value, SelectKBest)
                       and not isinstance(value, NLTKPreprocessor)
                       and not isinstance(value, WordNetLemmatizer)
                       and not key.endswith('steps')
                       and not key.endswith('transformer_list')}

    filtered_params = {key: (value if not isinstance(value, Pipeline) else 'Pipeline')
                       for key, value in filtered_params.items()}

    return filtered_params


def filtered_search_space(param_search_space):
    """Returns a new parameter dictionary with Pipeline values replaced by 'Pipeline'."""

    tuple_pipeline_string = lambda tpl: list(('Pipeline'
                                             if isinstance(value, Pipeline)
                                             else value
                                             for value in tpl))

    contains_pipeline = lambda value: isinstance(value, tuple) or isinstance(value, list) and any(isinstance(val, Pipeline) for val in value)

    filtered_params = {key: (tuple_pipeline_string(value)
                             if contains_pipeline(value)
                             else value)
                       for key, value in param_search_space.items()}

    return filtered_params


def print_fitting_report(pipeline, dt_fitting, x_train, y_train, min_estimators=None, max_estimators=None, score=None):
    """Prints the training report."""

    if min_estimators and max_estimators and score:

        # Generate the "OOB score" vs. "n_estimators" plot.
        xs, ys = zip(*score)
        plt.plot(xs, ys, label="Estimator")
        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB Score rate")
        plt.legend(loc="upper right")

        log.debug("Fitted {} data points.".format(len(y_train)))
        log.debug("Fitting and oob calculation done in {}".format(dt_fitting))

        plt.savefig('oob.png', transparent=True)
        #plt.show()

    else:
        log.debug("Fitted {} data points.".format(len(y_train)))
        log.debug("Fitting done in {}".format(dt_fitting))


def print_feature_importances_report(pipeline, dt_fitting, x_train, y_train, n_estimators=None, p_importances = None, p_indices = None):

    if n_estimators:
        # Print the feature ranking
        log.debug("Print the feature ranking...")
        for f in range(x_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, p_indices[f], p_importances[p_indices[f]]))
        
        # Plot the feature importances
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x_train.shape[1]), p_importances[p_indices], color="r", yerr=std[p_indices], align="center")
        plt.xticks(range(x_train.shape[1]), p_indices)
        plt.xlim([-1, x_train.shape[1]])

        log.debug("Fitted {} data points.".format(len(y_train)))
        log.debug("Fitting and feature importances done in {}".format(dt_fitting))

        plt.savefig('feature_importances.png', transparent=True)
        plt.show()

    else:
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

    df_results = DataFrame(cv_results)
    print(df_results)

    io.store_search_data(df_results)


def print_hyper_parameter_search_report(type, pipeline_configuration, dt_search, search):
    """Prints the search report for `type`."""

    print("Hyper parameter search report:")
    print_search_space(pipeline_configuration.parameters(type))

    print("Best score: %0.3f" % search.best_score_)

    print_best_parameters(search, pipeline_configuration.parameters(type))
    print_folds_results(search)

    log.debug(type + " search done in {}".format(dt_search))


def print_hyper_parameters(pipeline):
    """Prints the i.e. grid search report."""

    print("Hyper parameters:")
    pprint(filtered(pipeline.get_params()))
