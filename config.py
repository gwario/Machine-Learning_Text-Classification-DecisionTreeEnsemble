import pandas as pd
from numpy.random import RandomState
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectKBest
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
            #('feature_count_printer', FeatureCountPrinter(key+'_additional_data_vectorizer_pipeline')),
        ])

    def clf_extra_trees(self):
        return ExtraTreesClassifier(random_state=self.classifier_random_state, n_jobs=-1,
                                    min_impurity_split=1e-05,
                                    max_features='log2')

    def clf_random_forest(self):
        return RandomForestClassifier(random_state=self.classifier_random_state, n_jobs=-1,
                                      min_impurity_split=1e-05,
                                      max_features='log2')

    def union_pipeline(self, subpipelines):
        # Use FeatureUnion to combine the features
        return Pipeline([
            ('union', FeatureUnion(subpipelines, n_jobs=-1)),
            ('feature_count', FeatureCountPrinter('union')),
            #('select_mic', SelectKBest(mutual_info_classif, k=1000)),
            #('feature_count_mic', FeatureCountPrinter('kbest_mic')),
            ])

    def clf_pipeline(self):
        # Use FeatureUnion to combine the features
        return Pipeline([('select', SelectKBest(chi2)),
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
            ('abstractPosTokLemSyn', self.additional_data_vectorizer_pipeline('Tokens', (1, 3))),

            # 32588 features on 1616 samples
            ('title_word_ngrams', self.word_ngrams_pipeline('Title', (1, 3))),
            # 384864 features on 1616 samples
            ('abstract_word_ngrams', self.word_ngrams_pipeline('Abstract', (1, 3))),

            # 132175 features on 1616 samples
            ('title_char_ngrams', self.char_ngrams_pipeline('Title', (3, 9))),
            # 566240 features on 1616 samples
            ('abstract_char_ngrams', self.char_ngrams_pipeline('Abstract', (3, 9))),

            # 2213 features in 1616 samples
            ('term_vector', self.additional_data_vectorizer_pipeline('Terms', (1, 1))),
            # 2306 features in 1616 samples
            ('keyword_vector', self.additional_data_vectorizer_pipeline('Keywords', (1, 1))),

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
    pipeline_parameters_randomized_n_iter = 178 # space = 12320 / 6 = 2053
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    # Must be greater or equal to 2
    pipeline_parameters_randomized_n_splits = 3
    # To ensure some reproducibility
    pipeline_parameters_randomized_random_state = RandomState(654321)

    def binary_pipeline_parameters_randomized(self):

        return {
            'select__k': [10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,450,500,550,
                          600,650,700,750,800,850,900,950,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3500,
                          4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,12500,15000,17500,20000,
                          25000,50000,75000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,
                          600000,650000,700000,750000,800000,850000,900000,950000,1000000,1100000,1200000,1300000,
                          1400000,1485775, 'all'],
            'clf': [self.clf_extra_trees(), self.clf_random_forest()],
            'clf__max_depth': [12],#[6,7,8,9,10,12,13,14,15,16,17,18,19,20],
            'clf__max_leaf_nodes': [35],
            'clf__min_samples_leaf': [3],
            'clf__min_samples_split': [5],
            'clf__n_estimators': [420],#[1289,1290,1291,1292,1293,1294],  # Has to be > 25 for oob
        }

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
            'clf__criterion': 'gini',
            'clf__max_depth': 11,
            'clf__max_features': 'log2',
            'clf__max_leaf_nodes': 55,
            'clf__min_impurity_split': 1e-05,
            'clf__min_samples_leaf': 2,
            'clf__min_samples_split': 3,
            'clf__min_weight_fraction_leaf': 0.0,
            'clf__n_estimators': 1000, #  Has to be > 25 for oob
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
