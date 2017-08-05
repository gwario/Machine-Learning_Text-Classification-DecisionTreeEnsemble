import pandas as pd
from numpy.random import RandomState
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline

from extractor import ItemSelector, FeatureCountPrinter
from preprocessor import additional_data_tokenizer

__doc__ = """
Contains the configuration for binary and multiclass data.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

split_random_state = RandomState(123456)


class PipelineConfiguration:

    def __init__(self, dataset):
        self.dataset = dataset

    #########################################
    # Pipelines
    ###########
    def word_ngrams_pipeline(self, selector_key, ngram_range=(1,2)):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('vectorizer', TfidfVectorizer(strip_accents='unicode',
                                               analyzer='word',
                                               ngram_range=ngram_range,
                                               stop_words='english')),
                #('feature_count_printer', FeatureCountPrinter(selector_key+'_word_ngrams_pipeline')),
        ])

    def char_ngrams_pipeline(self, selector_key, ngram_range=(3,7)):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('vectorizer', TfidfVectorizer(strip_accents='unicode',
                                               analyzer='char_wb',
                                               ngram_range=ngram_range,
                                               stop_words='english')),
                #('feature_count_printer', FeatureCountPrinter(selector_key+'_char_ngrams_pipeline')),
        ])

    def additional_data_vectorizer_pipeline(self, key, ngram_range):
        return Pipeline([
            ('selector', ItemSelector(key=key)),
            ('vectorizer', TfidfVectorizer(tokenizer=additional_data_tokenizer,
                                           preprocessor=None,
                                           lowercase=False,
                                           ngram_range=ngram_range)),
            ('feature_count_printer', FeatureCountPrinter(key+'_additional_data_vectorizer_pipeline')),
        ])

    def clf_adabost_extra_trees(self):
        return AdaBoostClassifier(base_estimator=ExtraTreesClassifier(random_state=self.classifier_random_state,
                                                                      n_jobs=-1,
                                                                      min_impurity_split=1e-05,
                                                                      max_features='log2'),
                                  random_state=self.classifier_random_state)

    def clf_extra_trees(self):
        return ExtraTreesClassifier(random_state=self.classifier_random_state, n_jobs=-1,
                                    min_impurity_split=1e-05,
                                    max_features='log2')

    def clf_adabost_random_forest(self):
        return AdaBoostClassifier(base_estimator=RandomForestClassifier(random_state=self.classifier_random_state,
                                                                        n_jobs=-1,
                                                                        min_impurity_split=1e-05,
                                                                        max_features='log2'),
                                  random_state=self.classifier_random_state)

    def clf_random_forest(self):
        return RandomForestClassifier(random_state=self.classifier_random_state, n_jobs=-1,
                                      min_impurity_split=1e-05,
                                      max_features='log2')

    def union_pipeline(self, subpipelines):
        # Use FeatureUnion to combine the features
        return Pipeline([
            ('union', FeatureUnion(subpipelines, n_jobs=-1)),
            #('feature_count', FeatureCountPrinter('union')),
            ('select_mic', SelectKBest(chi2, k=4000)),
            #('feature_count_mic', FeatureCountPrinter('kbest_mic')),
            ])

    def clf_pipeline(self):
        # Use FeatureUnion to combine the features
        return Pipeline([#('select', SelectKBest(chi2)),
                         ('clf', self.clf_extra_trees()),
                         # ('clf', DummyClassifier()),
                         ])

    def binary_or_multi(self, binaryOption, multiOption):
        if self.dataset == 'binary':
            return binaryOption
        elif self.dataset == 'multi-class':
            return multiOption

    def pipelines(self):
        return self.binary_or_multi(self.binary_pipelines(), self.multiclass_pipelines())

    def parameters(self, type='config'):
        if type == 'config':
            return self.binary_or_multi(self.binary_pipeline_parameters(), self.multiclass_pipeline_parameters())
        elif type == 'grid':
            return self.binary_or_multi(self.binary_pipeline_parameters_grid(), self.multiclass_pipeline_parameters_grid())
        elif type == 'randomized':
            return self.binary_or_multi(self.binary_pipeline_parameters_randomized(), self.multiclass_pipeline_parameters_randomized())
        elif type == 'evolutionary':
            return self.binary_or_multi(self.binary_pipeline_parameters_evolutionary(), self.multiclass_pipeline_parameters_evolutionary())


    # ### Binary ####################################################
    def binary_pipelines(self):
        return (self.union_pipeline([
            # 365389 feature on 1616 samples
            ('abstractPosTokLemSyn', self.additional_data_vectorizer_pipeline('Tokens', (1, 2))),

            # 32588 features on 1616 samples
            #('title_word_ngrams', self.word_ngrams_pipeline('Title', (1, 3))),
            # 384864 features on 1616 samples
            #('abstract_word_ngrams', self.word_ngrams_pipeline('Abstract', (1, 3))),

            # 132175 features on 1616 samples
            #('title_char_ngrams', self.char_ngrams_pipeline('Title', (3, 9))),
            # 566240 features on 1616 samples
            #('abstract_char_ngrams', self.char_ngrams_pipeline('Abstract', (3, 9))),

            # 2213 features in 1616 samples
            #('term_vector', self.additional_data_vectorizer_pipeline('Terms', (1, 1))),
            # 2306 features in 1616 samples
            #('keyword_vector', self.additional_data_vectorizer_pipeline('Keywords', (1, 1))),

            # sum of all feature vectors is 1485775
        ]), self.clf_pipeline())

    # ### Multiclass ####################################################
    def multiclass_pipelines(self):
        return (self.union_pipeline([
            # Pipeline for pulling features from the articles's title
            # ('textWordCount', self.word_count_pipeline('Text')),
            ('textTokenizedAndLemmatized', self.additional_data_vectorizer_pipeline('Tokens', (1, 3))),
            ('word_ngrams', self.word_ngrams_pipeline('Text')),
            ('char_ngrams', self.char_ngrams_pipeline('Text')),
        ]), self.clf_pipeline())

    classifier_random_state = RandomState(162534)


    ##########################################
    # Hyper-parameter search configuration
    ######################################


    ###################################################################
    # This set of parameters is used when --hp randomized was specified.
    ############
    # The parameter space must be larger than or equal to n_iter
    pipeline_parameters_randomized_n_iter = 1 # space = 12320 / 6 = 2053
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    # Must be greater or equal to 2
    pipeline_parameters_randomized_n_splits = 50
    # To ensure some reproducibility
    pipeline_parameters_randomized_random_state = RandomState(654321)

    def binary_pipeline_parameters_randomized(self):
        return {
            'clf': [self.clf_extra_trees()],
            'clf__n_estimators': [1000],
        }

        """
        all et
        return {
            'select__k': [6000,10000,300000,411500,455500],
            'clf': [self.clf_extra_trees()],
            'clf__max_depth': [12,21,25,27,35,45],
            'clf__max_leaf_nodes': [40,50,None],
            'clf__min_samples_leaf': [1,2,3],
            'clf__min_samples_split': [2,5,8,10,12],
            'clf__n_estimators': [1100, 1600],
        }
        """
        """
        all rf
        return {
            'select__k': [6000,10000,411500, 455500],
            'clf': [self.clf_random_forest()],
            'clf__max_depth': [11],
            'clf__max_leaf_nodes': [60,70,None],
            'clf__min_samples_leaf': [1,2],
            'clf__min_samples_split': [2,3,],
            'clf__n_estimators': [1100, 1600],
        }
        """
        """ backup
        return {
            'select__k': [10000,411500],#, 407000, 411500, 416000, 418500, 427000, 427500, 428500, 434000, 451000, 455500, ],
            'clf': [self.clf_extra_trees(), self.clf_random_forest()],
            'clf__max_depth': [12,21,25,27,35,45],
            'clf__max_leaf_nodes': [40,70,None],
            'clf__min_samples_leaf': [1,2,3],
            'clf__min_samples_split': [2,4,5,8,10,12],
            'clf__n_estimators': [1600],#et: [1100],  # Has to be > 25 for oob
        }
        """
    def multiclass_pipeline_parameters_randomized(self):
        return {
            'select__k': [1000],
            'clf__max_depth': (2, 5, 10, 20),
            'clf__n_estimators': (10, 20, 50, 80, 300),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', (1,2)))
        }


    # This custom set of parameters is used when --hp config was specified.
    def binary_pipeline_parameters(self):
        return {
            'select__score_func': mutual_info_classif,
            'select__k': 4000,
            'clf': self.clf_extra_trees(),
            'clf__n_estimators': 1531,
        }

    def multiclass_pipeline_parameters(self):
        return {
            'select__k': 750,
            'clf__max_depth': 10,
            'clf__n_estimators': 1200,
        }

    # This set of parameters is used when --hp evolutionary was specified.
    pipeline_parameters_evolutionary_random_seed = 32135
    pipeline_parameters_evolutionary_n_splits = 2
    pipeline_parameters_evolutionary_population_size = 50
    pipeline_parameters_evolutionary_gene_mutation_prob = 0.10
    pipeline_parameters_evolutionary_gene_crossover_prob = 0.5
    pipeline_parameters_evolutionary_tournament_size = 3,
    pipeline_parameters_evolutionary_generations_number = 5

    def binary_pipeline_parameters_evolutionary(self):
        return {
            'clf__max_depth': [2, 10, 40],
            'clf__n_estimators': [10, 80],
            'union__abstractWordCount': [None],
            'union__abstractPosTokLemSyn': [None],
        }

    def multiclass_pipeline_parameters_evolutionary(self):
        return {
            'select__k': [1000],
            'clf__max_depth': (2, 10, 40),
            'clf__n_estimators': (10, 80, 300),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', (1,2)))
        }

    # This set of parameters is used when --hp grid was specified.
    # http://scikit-learn.org/stable/modules/grid_search.html#grid-search
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    pipeline_parameters_grid_n_splits = 2
    def binary_pipeline_parameters_grid(self):
        return {
            'clf__max_depth': (2, 5),
            'clf__n_estimators': (10, 80),
            'union__abstractWordCount': (None, self.word_ngrams_pipeline('Abstract', (1,1))),
            'union__abstractPosTokLemSyn': (None, self.additional_data_vectorizer_pipeline('Tokens', (1,2))),
        }

    def multiclass_pipeline_parameters_grid(self):
        return {
            'select__k': [1000],
            'clf__max_depth': (2, 5),
            'clf__n_estimators': (10, 80),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', (1,2))),
        }
