import numpy as np
import file_io as io
from preprocessor import NLTKPreprocessor
import argparse
import logging as log
import pandas as pd

def get_args(args_parser):
    """Parses and returns the command line arguments."""

    args_parser.add_argument('--data', metavar='FILE', type=argparse.FileType('r'), help='Data to be preprocessed.')
    args_parser.add_argument('--filename', metavar='FILE', type=argparse.FileType('r'), help='The filename for the new processed data.')
    return args_parser.parse_args()


nltk_preprocessor = NLTKPreprocessor()

def get_key(dataset_type):
	if dataset_type == 'binary':
		return 'Abstract'
	elif dataset_type == 'multi-class':
		return 'Text'

def preprocess(dataset, key):
	log.info("Preprocessing data")

	tokens = nltk_preprocessor.transform(dataset[key])
	joined = [' '.join(t) for t in tokens]
	# print(joined)
	# print(dataset)
	series = pd.Series(joined, index=dataset.index)
	dataset['Tokens'] = series
	# return np.insert(dataset, dataset.shape[1], joined, axis=1)
	return dataset

if __name__ == '__main__':
	# Display progress logs on stdout
	log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	args = get_args(parser)
	dataset_type = io.get_data_set(args.data)
	x, y = io.load_data(args.data)
	new_data = preprocess(x, get_key(dataset_type))
	new_data = pd.concat([new_data, y], axis=1)
	io.save_data(new_data, args.filename)
	log.info("Saved to file: {}".format(filename))
