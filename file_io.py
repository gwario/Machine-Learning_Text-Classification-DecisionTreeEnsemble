import logging as log
import os.path

import pandas as pd
from sklearn.externals import joblib

__doc__ = """
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

    if os.path.exists(data_file.name + ".add"):
        additional_data = load_additional_data(data_file.name)
        df = pd.merge(df, additional_data, how='inner', on='Id')

    df = df.fillna('')

    article_columns = [column for column in df if column in ['Id', 'Title', 'Abstract', 'Text', 'Keywords', 'Terms', 'Tokens']]

    return df.loc[:, article_columns], df.loc[:, 'Category'] if 'Category' in df else None


def get_data_set(data_set_file):
    """Returns the data-set identification string. One of 'binary', 'multi-class' or, in case of an invalid datas-et,
    'unknown_data_set'"""

    data_set_file.seek(0)
    first_line = next(data_set_file)
    data_set_file.seek(0)

    if 'Title' in first_line and 'Abstract' in first_line:
        log.info("Recognised data-set: binary")
        return 'binary'

    elif 'Text' in first_line:
        log.info("Recognised data-set: multi-class")
        return 'multi-class'

    else:
        log.error("Invalid data-set!")
        exit(1)


def load_model(model_file):
    """Loads and returns the pipeline model from model_file."""

    log.debug("Loading model from {}".format(model_file))
    model = joblib.load(model_file)
    return model[0], model[1]


def save_model(fu_pl, clf_pl, model_filename):
    """Saves the pipeline model to model_filename."""

    joblib.dump([fu_pl, clf_pl], model_filename)
    print("Model saved as {}".format(model_filename))


def save_data(dataset, dataset_filename):
    """Saves the dataset to dataset_filename."""

    dataset.to_csv(dataset_filename, sep=',', index=False, encoding='utf-8')
    print("Dataset saved as {}".format(dataset_filename))


def save_prediction(prediction, prediction_filename):
    """Saves the prediction to prediction_filename."""

    prediction.to_csv(prediction_filename, sep=',', index=False, encoding='utf-8')
    print("Prediction saved as {}".format(prediction_filename))


def load_additional_data(data_file):
    """Loads the keywords data from <data_file>.add."""

    log.debug("Loading data...")
    tp = pd.read_csv(data_file + ".add", iterator=True, chunksize=1000)
    df = pd.concat(tp, ignore_index=True)

    return df


def save_additional_data(data, filename):
    """Saves the data to <filename>.add."""
    data.to_csv(filename + ".add", sep=',', index=False, encoding='utf-8')
    print("Additional data saved as {}".format(filename))


def store_oob_score_data(params, oob_errors):
    """Stores the oob error data in to a oob.csv"""
    filename = "oob.csv"
    data = pd.DataFrame(oob_errors)
    #print(data)

    data.to_csv(filename, sep=',', index=False, encoding='utf-8')
    print("OOB Score saved as {}".format(filename))


def store_search_data(data):
    """Stores the oob error data in to a oob.csv"""
    filename = "search_results.csv"

    data.to_csv(filename, sep=',', index=False, encoding='utf-8')
    print("Search results saved as {}".format(filename))