from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from extractor import Printer, ItemSelector
from preprocessor import NLTKPreprocessor

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

'''Contains the configuration for binary and multiclass data.'''

split_random_state = 123456
n_iter_search = 2

# Pipelines

def word_count_pipeline(selector_key):
    return Pipeline([
            ('selector', ItemSelector(key=selector_key)),        # ('printer', Printer()),
            ('count', CountVectorizer()),                   # ('printer', Printer()),
        ])

def tokenized_and_lemmatized_pipeline(selector_key): 
    return Pipeline([
            ('selector', ItemSelector(key=selector_key)),
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ])

binary_pipeline = Pipeline([
    # Use FeatureUnion to combine the features
    ('union', FeatureUnion([

        # Pipeline for pulling features from the articles's title
        ('titleWordCount', word_count_pipeline('Title')),
        ('abstractWordCount', word_count_pipeline('Abstract')),
        ('tokenizedAndLemmatized', tokenized_and_lemmatized_pipeline('Abstract')),
        
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
    ],
    )),
    # ('printer', Printer()),
    ('clf', RandomForestClassifier()),
])

# ### Binary ####################################################
# This parameters grid is used to find the best parameter configuration for the
# pipeline when --grid was specified.
# http://scikit-learn.org/stable/modules/grid_search.html#grid-search
# TODO add some variation of the default parameters
binary_pipeline_parameters_grid = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'clf__alpha': (0.00001, 0.000001),
    # 'clf__n_iter': (10, 50, 80),
    'clf__max_depth': (2, 5),
    'clf__n_estimators': (10, 80),
    'union__abstractWordCount': (None, word_count_pipeline('Abstract')),
    'union__tokenizedAndLemmatized': (None, tokenized_and_lemmatized_pipeline('Abstract')),
}

binary_pipeline_parameters_randomized = {
  # "max_features": sp_randint(1, 11),
  # "min_samples_split": sp_randint(1, 11),
  # "min_samples_leaf": sp_randint(1, 11),
  # "bootstrap": [True, False],
  # "criterion": ["gini", "entropy"]
    'clf__max_depth': (2, 5),
    'clf__n_estimators': (10, 80),
    'union__abstractWordCount': (None, word_count_pipeline('Abstract')),
    'union__tokenizedAndLemmatized': (None, tokenized_and_lemmatized_pipeline('Abstract')),
}

# This custom set of parameters is used when --grid was NOT specified.
# TODO add all the default parameters here
binary_pipeline_parameters = {
    'clf__max_depth': 8,
    'clf__n_estimators': 10,
}

# ### Multiclass ####################################################
# This parameters grid is used to find the best parameter configuration for the
# pipeline when --grid was specified.
# http://scikit-learn.org/stable/modules/grid_search.html#grid-search
# TODO add some variation of the default parameters
multiclass_pipeline_parameters_grid = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'clf__alpha': (0.00001, 0.000001),
    # 'clf__n_iter': (10, 50, 80),
    'clf__max_depth': (2, 5),
    'clf__n_estimators': (10, 80),
    'union__tokenizedAndLemmatized': (None, tokenized_and_lemmatized_pipeline('Text')),
}

# This custom set of parameters is used when --grid was NOT specified.
# TODO add all the default parameters here
multiclass_pipeline_parameters = {
    'clf__max_depth': 8,
    'clf__n_estimators': 10,
}

multiclass_pipeline = Pipeline([
    # Use FeatureUnion to combine the features
    ('union', FeatureUnion([

        # Pipeline for pulling features from the articles's title
        ('textWordCount', word_count_pipeline('Text')),
        ('tokenizedAndLemmatized', tokenized_and_lemmatized_pipeline('Text')),
        #TODO add your feature vectors here
        # Pipeline for pulling features from the articles's abstract
        #('myfeature', Pipeline([
        #    ('selector', ItemSelector(key='abstract')),
        #    ('hasher', HashingVectorizer()),
        #])),
    ],
    )),
    # ('printer', Printer()),
    ('clf', RandomForestClassifier()),
])
