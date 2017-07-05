#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import argparse
import logging as log

import pandas as pd
from Bio import Entrez

import file_io as io

__doc__ = """
Fetches the kewords and mesh-terms for all articles 
"""
__author__ = "Mario Gastegger, Alex Heemann, Desislava Velinova"
__copyright__ = "Copyright 2017, "+__author__
__license__ = "FreeBSD License"
__version__ = "1.1"
__status__ = "Production"


# Display progress logs on stdout
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

Entrez.email = 'gastegm@informatik.hu-berlin.de'


def get_args(args_parser):
    """Parses and returns the command line arguments."""

    args_parser.add_argument('--data', metavar='FILE', type=argparse.FileType('r'),
                             help='''The data to get keywords and terms for. The terms are saved to disk.
                             <data>.add''')

    return args_parser.parse_args()


def check_mode_of_operation(arguments):
    """Checks the mode of operation."""

    if arguments.data:
        log.debug("Valid mode of operation")
    else:
        print("Invalid mode of operation!")
        parser.print_help()
        exit(1)


def search(query):
    handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='1', retmode='xml', term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
    results = Entrez.read(handle)
    return results


def get_keywords_terms(title):

    results = search(title)
    log.debug("Looking for article with title '{}'".format(title))
    id_list = results['IdList']

    keywords = []
    mesh_terms = []
    if len(id_list) > 0:

        papers = fetch_details(id_list)

        if len(papers['PubmedArticle']) > 0:

            medline_citations = papers['PubmedArticle'][0]['MedlineCitation']

            # MeSH terms
            if 'MeshHeadingList' in medline_citations\
                    and len(medline_citations['MeshHeadingList']) > 0:
                mesh_terms += [str(t['DescriptorName']) for t in medline_citations['MeshHeadingList']]
                log.debug("Got terms.")
            else:
                log.warning("MeSH terms not available.")

            # Keywords
            if 'KeywordList' in medline_citations\
                    and len(medline_citations['KeywordList']) > 0:
                keywords += [str(kw) for kw in medline_citations['KeywordList'][0]]
                log.debug("Got keywords.")
            else:
                log.warning("Keywords not available.")
        else:
            log.warning("Not an article.")
    else:
        log.warning("Article not available.")

    return keywords, mesh_terms


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = get_args(parser)
    log.debug("Commandline arguments: {}".format(args))

    check_mode_of_operation(args)

    data, _ = io.load_data(args.data)

    if io.get_data_set(args.data) == 'binary':

        keywords = []
        terms = []

        log.info("Fetching keywords an terms...")

        kw_cnt = 0
        t_cnt = 0
        for index, row in data.iterrows():

            #if index > 4:
            #    keywords.append(" ")
            #    terms.append(" ")
            #else:
            kws, ts = get_keywords_terms(row['Title'])
            keywords.append(';'.join(str(x) for x in kws))
            terms.append(';'.join(str(x) for x in ts))

            if len(kws) > 0:
                kw_cnt += 1
            if len(ts) > 0:
                t_cnt += 1

            log.info("Got {} keywords and {} terms for {} articles.".format(kw_cnt, t_cnt, index+1))

        data = data.loc[:, ['Id']]
        data = data.assign(Terms=pd.Series(terms).values)
        data = data.assign(Keywords=pd.Series(keywords).values)
        io.save_additional_data(data, args.data.name)

    else:
        print("Keywords and terms lookup only possible for binary data.")

