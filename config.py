import pandas as pd
from numpy.random import RandomState
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from extractor import Printer, ItemSelector
from preprocessor import NLTKPreprocessor

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

class PipelineConfiguration:

    def __init__(self, dataset):
        self.dataset = dataset

    #########################################
    # Pipelines
    ###########
    def word_count_pipeline(self, selector_key):
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),        # ('printer', Printer()),
                ('count', CountVectorizer()),                   # ('printer', Printer()),
            ])


    def tokenized_and_lemmatized_pipeline(self, selector_key): 
        return Pipeline([
                ('selector', ItemSelector(key=selector_key)),
                ('preprocessor', NLTKPreprocessor()),
                ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ])

    def union_pipeline(self, subpipelines):
        # Use FeatureUnion to combine the features
        return Pipeline([
            ('union', FeatureUnion(subpipelines)),
            ('clf', RandomForestClassifier(random_state=self.classifier_random_state, n_jobs=-1)),    
            ])

    def getBinaryOrMulti(self, binaryOption, multiOption):
        if self.dataset == 'binary':
            return binaryOption
        elif self.dataset == 'multi-class':
            return multiOption

    def getPipeline(self):
        return getBinaryOrMulti(self.binary_pipeline, self.multiclass_pipeline)

    def getParameters(self, type='config'):
        if type == 'config':
            return self.getBinaryOrMulti(self.binary_pipeline_parameters, self.multiclass_pipeline_parameters)
        elif type == 'grid':
            return self.getBinaryOrMulti(self.binary_pipeline_parameters_grid, self.multiclass_pipeline_parameters_grid)
        elif type == 'randomized':
            return self.getBinaryOrMulti(self.binary_pipeline_parameters_randomized, self.multiclass_pipeline_parameters_randomized)

    # ### Binary ####################################################
    binary_pipeline = self.union_pipeline([

            # Pipeline for pulling features from the articles's title
            ('titleWordCount', self.word_count_pipeline('Title')),
            ('abstractWordCount', self.word_count_pipeline('Abstract')),
            ('tokenizedAndLemmatized', self.tokenized_and_lemmatized_pipeline('Abstract')),

            #('title', Pipeline([
            # ('selector', ItemSelector(key='Title')),
            # ('tfidf', TfidfVectorizer()),
            #])),

            #TODO add your feature vectors here
            # Pipeline for pulling features from the articles's abstract
            #('myfeature', Pipeline([
            #    ('selector', ItemSelector(key='abstract')),
            #    ('hasher', HashingVectorizer()),
            #])),
        ])

    # ### Multiclass ####################################################
    multiclass_pipeline = self.union_pipeline([        
            # Pipeline for pulling features from the articles's title
            ('textWordCount', self.word_count_pipeline('Text')),
            ('tokenizedAndLemmatized', self.tokenized_and_lemmatized_pipeline('Text')),
            #TODO add your feature vectors here
            # Pipeline for pulling features from the articles's abstract
            #('myfeature', Pipeline([
            #    ('selector', ItemSelector(key='abstract')),
            #    ('hasher', HashingVectorizer()),
            #])),
        ])

    ##########################################
    # Hyper-parameter search configuration
    ######################################

    classifier_random_state = RandomState(162534)
    split_random_state = RandomState(123456)

    # This custom set of parameters is used when --hp config was specified.
    binary_pipeline_parameters = {
        'clf__max_depth': 8,
        'clf__n_estimators': 10,
    }
    multiclass_pipeline_parameters = {
        'clf__max_depth': 8,
        'clf__n_estimators': 10,
    }

    # This set of parameters is used when --hp randomized was specified.
    # The parameter space must be larger than or equal to n_iter
    pipeline_parameters_randomized_n_iter = 2
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    # Must be greater or equal to 2
    pipeline_parameters_randomized_n_splits = 2
    # To ensure some reproducibility
    pipeline_parameters_randomized_random_state = RandomState(654321)
    binary_pipeline_parameters_randomized = {
        'clf__max_depth': (2, 5, 10, 20),
        'clf__n_estimators': (10, 20, 50, 80, 300),
        'union__abstractWordCount': (None, self.word_count_pipeline('Abstract')),
        'union__tokenizedAndLemmatized': (None, self.tokenized_and_lemmatized_pipeline('Abstract')),
    }
    multiclass_pipeline_parameters_randomized = {
        'clf__max_depth': (2, 5, 10, 20),
        'clf__n_estimators': (10, 20, 50, 80, 300),
        'union__tokenizedAndLemmatized': (None, self.tokenized_and_lemmatized_pipeline('Text')),
    }

    # This parameters grid is used to find the best parameter configuration for the pipeline when --hp grid was specified.
    # http://scikit-learn.org/stable/modules/grid_search.html#grid-search
    # The default is to cross-validate with 3 folds, this takes a considerable amount of time
    pipeline_parameters_grid_n_splits = 2
    binary_pipeline_parameters_grid = {
        'clf__max_depth': (2, 5),
        'clf__n_estimators': (10, 80),
        'union__abstractWordCount': (None, self.word_count_pipeline('Abstract')),
        'union__tokenizedAndLemmatized': (None, self.tokenized_and_lemmatized_pipeline('Abstract')),
    }
    multiclass_pipeline_parameters_grid = {
        'clf__max_depth': (2, 5),
        'clf__n_estimators': (10, 80),
        'union__tokenizedAndLemmatized': (None, self.tokenized_and_lemmatized_pipeline('Text')),
    }

