import pandas as pd
from numpy.random import RandomState
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif, SelectFromModel

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from extractor import Printer, ItemSelector, FeatureCountPrinter
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
    def word_count_pipeline(self, selector_key, k=4000):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('count', CountVectorizer()),
                ('select', SelectKBest(chi2, k=k)),
                #('feature_count_printer', FeatureCountPrinter(selector_key+'_word_count_pipeline')),
            ])

    def word_ngrams_pipeline(self, selector_key):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('vectorizer', TfidfVectorizer(strip_accents='unicode',
                                               analyzer='word',
                                               ngram_range=(1, 2),
                                               stop_words='english')),
                ('select', SelectKBest(chi2)),
                #('select', SelectKBest(mutual_info_classif, k=1500)),
                #('feature_count_printer', FeatureCountPrinter(selector_key+'_word_ngrams_pipeline')),
        ])

    def char_ngrams_pipeline(self, selector_key):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('vectorizer', TfidfVectorizer(strip_accents='unicode',
                                               analyzer='char',
                                               ngram_range=(3, 7),
                                               stop_words='english')),
                ('select', SelectKBest(chi2)),
                #('select_chi2', SelectKBest(mutual_info_classif, k=1500)),
                #('feature_count_printer', FeatureCountPrinter(selector_key+'_char_ngrams_pipeline')),
        ])

    def additional_data_vectorizer_pipeline(self, key, k, ngram_range):
        return Pipeline([
            ('selector', ItemSelector(key=key)),
            ('vectorizer', TfidfVectorizer(tokenizer=additional_data_tokenizer, preprocessor=None, lowercase=False, ngram_range=ngram_range)),
            #('feature_count_printer', FeatureCountPrinter(key+'_additional_data_vectorizer_pipeline')),
            ('select', SelectKBest(chi2, k=k)),
            #('select_mutinfcls', SelectKBest(mutual_info_classif)),
        ])

    def union_pipeline(self, subpipelines):
        # Use FeatureUnion to combine the features
        return Pipeline([
            ('union', FeatureUnion(subpipelines, transformer_weights={'abstractPosTokLemSyn': 12, 'term_vector': 25, 'keyword_vector': 19, 'titleWordCount': 16})),
            #('feature_count', FeatureCountPrinter('union')),
            #('select_chi2', SelectKBest(chi2, k=1000)),
            #('feature_count_chi2', FeatureCountPrinter('kbest_chi2')),
            #('select_mic', SelectKBest(mutual_info_classif, k=1000)),
            #('feature_count_mic', FeatureCountPrinter('kbest_mic')),
            # ('printer', Printer()),
            ('clf', ExtraTreesClassifier(random_state=self.classifier_random_state, n_jobs=-1, min_impurity_split=1e-05, max_features='log2')),
            # ('clf', DummyClassifier()),    
            ])

    def binary_or_multi(self, binaryOption, multiOption):
        if self.dataset == 'binary':
            return binaryOption
        elif self.dataset == 'multi-class':
            return multiOption

    def pipeline(self):
        return self.binary_or_multi(self.binary_pipeline(), self.multiclass_pipeline())

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
    def binary_pipeline(self):
        return self.union_pipeline([
            # Pipeline for pulling features from the articles's title
            ('titleWordCount', self.word_count_pipeline('Title', 2000)),
            #('abstractWordCount', self.word_count_pipeline('Abstract')),
            ('abstractPosTokLemSyn', self.additional_data_vectorizer_pipeline('Tokens', 20000, (1,2))),
            #('word_ngrams', self.word_ngrams_pipeline('Abstract')),
            #('char_ngrams', self.char_ngrams_pipeline('Abstract')),
            ('term_vector', self.additional_data_vectorizer_pipeline('Terms', 1650, (1,1))),
            ('keyword_vector', self.additional_data_vectorizer_pipeline('Keywords', 1150, (1,1))),
        ])

    # ### Multiclass ####################################################
    def multiclass_pipeline(self):
        return self.union_pipeline([
            # Pipeline for pulling features from the articles's title
            # ('textWordCount', self.word_count_pipeline('Text')),
            # ('textTokenizedAndLemmatized', self.tokenized_and_lemmatized_pipeline('Text')),
            # ('word_ngrams', self.word_ngrams_pipeline('Text')),
            ('char_ngrams', self.char_ngrams_pipeline('Text')),
        ])

    classifier_random_state = RandomState(162534)


    ##########################################
    # Hyper-parameter search configuration
    ######################################

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
	    'union__titleWordCount__select_chi2__k': 4000,
            'union__abstractPosTokLemSyn__select_chi2__k':	42000,
            'union__term_vector__select_chi2__k':     		1650,
            'union__keyword_vector__select_chi2__k':		1150,

        }

    def multiclass_pipeline_parameters(self):
        return {
            'clf__max_depth': 8,
            'clf__n_estimators': 10,
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
            'clf__max_depth': (2, 10, 40),
            'clf__n_estimators': (10, 80, 300),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', 30000, (1,2)))
        }

    ###################################################################
    # This search space takes about 12 minutes to search
    #
    # This set of parameters is used when --hp randomized was specified.
    ############
    # The parameter space must be larger than or equal to n_iter
    pipeline_parameters_randomized_n_iter = 80
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    # Must be greater or equal to 2
    pipeline_parameters_randomized_n_splits = 8
    # To ensure some reproducibility
    pipeline_parameters_randomized_random_state = RandomState(654321)

    def binary_pipeline_parameters_randomized(self):

        return {
            'clf__max_depth': [14, 15],
            'clf__max_leaf_nodes': [35, 40, 65],
            'clf__min_samples_leaf': [1],
            'clf__min_samples_split': [3, 4, 5],
            'clf__n_estimators': [1289,1290,1291,1292,1293,1294],  # Has to be > 25 for oob
        }

    def multiclass_pipeline_parameters_randomized(self):
        return {
            'clf__max_depth': (2, 5, 10, 20),
            'clf__n_estimators': (10, 20, 50, 80, 300),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', 30000, (1,2)))
        }


    # This parameters grid is used to find the best parameter configuration for the pipeline when --hp grid was specified.
    # http://scikit-learn.org/stable/modules/grid_search.html#grid-search
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    pipeline_parameters_grid_n_splits = 2
    def binary_pipeline_parameters_grid(self):
        return {
            'clf__max_depth': (2, 5),
            'clf__n_estimators': (10, 80),
            'union__abstractWordCount': (None, self.word_count_pipeline('Abstract')),
            'union__abstractPosTokLemSyn': (None, self.additional_data_vectorizer_pipeline('Tokens', 41400, (1,2))),
        }

    def multiclass_pipeline_parameters_grid(self):
        return {
            'clf__max_depth': (2, 5),
            'clf__n_estimators': (10, 80),
            'union__textTokenizedAndLemmatized': (None, self.additional_data_vectorizer_pipeline('Tokens', 30000, (1,2))),
        }
