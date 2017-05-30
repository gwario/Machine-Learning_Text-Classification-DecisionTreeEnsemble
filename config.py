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
}

# This custom set of parameters is used when --grid was NOT specified.
# TODO add all the default parameters here
binary_pipeline_parameters = {
    'clf__max_depth': 8,
    'clf__n_estimators': 10,
}

binary_pipeline = Pipeline([
    # Use FeatureUnion to combine the features
    ('union', FeatureUnion([

        # Pipeline for pulling features from the articles's title
        ('titleWordCount', Pipeline([
            ('selector', ItemSelector(key='Title')),        # ('printer', Printer()),
            ('count', CountVectorizer()),                   # ('printer', Printer()),
        ])),
        ('abstractWordCount', Pipeline([
            ('selector', ItemSelector(key='Abstract')),     # ('printer', Printer()),
            ('count', CountVectorizer()),                   # ('printer', Printer()),
        ])),
        ('tokenizedAndLemmatized', Pipeline([
            ('selector', ItemSelector(key='Abstract')),
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ])),
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
        ('textWordCount', Pipeline([
            ('selector', ItemSelector(key='Text')),         # ('printer', Printer()),
            ('count', CountVectorizer()),                   # ('printer', Printer()),
        ])),
        ('tokenizedAndLemmatized', Pipeline([
            ('selector', ItemSelector(key='Text')),
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ])),
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