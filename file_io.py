import logging as log
import pandas as pd
from sklearn.externals import joblib

"""
Contains code to load and save data.
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


def load_data(data_file):
    """Loads the input data from data_file."""

    log.debug("Loading data...")
    # gives TextFileReader, which is iterable with chunks of 1000 rows.
    tp = pd.read_csv(data_file, iterator=True, chunksize=1000)
    # df is DataFrame. If errors, do `list(tp)` instead of `tp`
    df = pd.concat(tp, ignore_index=True)

    article_columns = [column for column in df if column in ['Id', 'Title', 'Abstract', 'Text']]

    return df.loc[:, article_columns], df.loc[:, 'Category'] if 'Category' in df else None


def load_model(model_file):
    """Loads and returns the pipeline model from model_file."""

    log.debug("Loading model from {}".format(model_file))
    return joblib.load(model_file)


def save_model(pipeline, model_filename):
    """Saves the pipeline model to model_filename."""

    joblib.dump(pipeline, model_filename)
    print("Model saved as {}\n".format(model_filename))


def save_prediction(prediction, prediction_filename):
    """Saves the prediction to prediction_filename."""

    prediction.to_csv(prediction_filename, sep=',', index=False, encoding='utf-8')
    print("Prediction saved as {}\n".format(prediction_filename))
