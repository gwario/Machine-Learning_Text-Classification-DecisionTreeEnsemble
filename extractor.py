"""
Contains feature extractors.

"""
from pprint import pprint
import logging as log
from sklearn.base import BaseEstimator, TransformerMixin


class Printer(BaseEstimator, TransformerMixin):
    """Prints the input."""
    def fit(self, x, y=None):
        return self

    def transform(self, input):
        print(input)
        return input

class FeatureCountPrinter(BaseEstimator, TransformerMixin):
    """Prints the input."""
    def __init__(self, vector_name=None):
        self.vector_name = vector_name

    def fit(self, x, y=None):
        return self

    def transform(self, input):

        (row_cnt, column_cnt) = input.shape

        if self.vector_name:
            log.info("Vector {} has {} features.".format(self.vector_name, column_cnt))
        else:
            log.info("Vector has {} features.".format(column_cnt))

        return input


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
