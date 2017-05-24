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


def print_training_report(pipeline, dt_fitting, dt_validation, articles, categories_true, categories_predicted):
    """Prints the training report."""

    # TODO Improve visualization of the cross-validation report i.e. add useful metrics from http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

    print("Cross-validation report:")
    print("Hyper parameters:")
    pprint(filtered(pipeline.get_params()))
    print("")
    print(classification_report(categories_true, categories_predicted))
    print("Fitting done in {}".format(dt_fitting))
    print("Cross-validation done in {}\n".format(dt_validation))


def print_prediction_report(pipeline, dt_predict, result):
    """Prints the classification result."""

    # TODO Improve visualization of the result report i.e. statistics on how many documents per class...

    print("Classification report:\n")
    print(result)
    print("\n")
    print("Hyper parameters:\n")
    pprint(filtered(pipeline.get_params()))
    print("Done in {}\n".format(dt_predict))


def print_hyper_parameter_search_report(pipeline, dt_search, parameter_grid, best_score, best_parameters):
    """Prints the i.e. grid search report."""

    print("Hyper parameter search report:")
    print("Parameter grid:")
    pprint(parameter_grid)
    print("Best score: %0.3f" % best_score)
    print("Best parameters set:")
    for param_name in sorted(parameter_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Grid search done in {}\n".format(dt_search))
