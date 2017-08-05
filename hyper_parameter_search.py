from datetime import datetime
import logging as log

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV

from config import PipelineConfiguration
import report as rp

__doc__ = """
Contains code to handle hyper-parameter optimization.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


def get_search(type, pipeline_configuration, hp_metric):
    fu_pl, clf_pl = pipeline_configuration.pipelines()
    if type == 'grid':
      return GridSearchCV(clf_pl,
                          pipeline_configuration.parameters('grid'), 
                          scoring=hp_metric,
                          cv=pipeline_configuration.pipeline_parameters_grid_n_splits,
                          refit=False,
                          n_jobs=-1, 
                          verbose=1), fu_pl
    elif type == 'randomized':
      return RandomizedSearchCV(clf_pl,
                                pipeline_configuration.parameters('randomized'),
                                scoring=hp_metric,
                                random_state=pipeline_configuration.pipeline_parameters_randomized_random_state,
                                n_iter=pipeline_configuration.pipeline_parameters_randomized_n_iter,
                                cv=pipeline_configuration.pipeline_parameters_randomized_n_splits,
                                refit=False,
                                n_jobs=3,
                                #pre_dispatch=6,
                                verbose=1), fu_pl
    elif type == 'evolutionary':
      from random import seed
      seed(pipeline_configuration.pipeline_parameters_evolutionary_random_seed)
      return EvolutionaryAlgorithmSearchCV(clf_pl,
                                           pipeline_configuration.parameters('evolutionary'), 
                                           scoring=hp_metric,
                                           cv=pipeline_configuration.pipeline_parameters_evolutionary_n_splits,
                                           population_size=pipeline_configuration.pipeline_parameters_evolutionary_population_size,
                                           gene_mutation_prob=pipeline_configuration.pipeline_parameters_evolutionary_gene_mutation_prob,
                                           gene_crossover_prob=pipeline_configuration.pipeline_parameters_evolutionary_gene_crossover_prob,
                                           tournament_size=pipeline_configuration.pipeline_parameters_evolutionary_tournament_size,
                                           generations_number=pipeline_configuration.pipeline_parameters_evolutionary_generations_number,
                                           refit=False,
                                           n_jobs=-1, 
                                           verbose=1), fu_pl


def get_result(type, pipeline_configuration, hp_metric, x, y):
    """ Finds and returns the best parameters for both the feature extraction and the classification.
    Changing the grid increases processing time in a combinatorial way."""

    log.debug("Performing {} search, optimizing {} score...".format(type, hp_metric))
    search, fu_pl = get_search(type, pipeline_configuration, hp_metric)

    log.debug("Generating feature vector...")
    t0 = datetime.now()
    x = fu_pl.fit_transform(x, y)
    log.info("Generated vector of {} features in {} from {} samples.".format(x.shape[1], datetime.now() - t0, x.shape[0]))

    t0 = datetime.now()
    search.fit(x, y)
    dt = datetime.now() - t0

    best_parameters = search.best_params_

    rp.print_hyper_parameter_search_report(type, pipeline_configuration, dt, search)

    return best_parameters


def get_optimized_parameters(type, pipeline_configuration, x_train, y_train, hp_metric):
    log.info("Using {} search on the training set to select the best model (hyper-parameters)...".format(type))

    return get_result(type, pipeline_configuration, hp_metric, x_train, y_train)
