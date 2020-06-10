import re
import json
import pickle
import logging
import time
import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch

from utils.data_utils import load_pkl_file


def connect_es(host='localhost', port=9200):
    '''
    Instantiate and return a Python ElasticSearch object

    Args:
        host (str)
        port (int)
    
    Returns:
        es (elasticsearch.client.Elasticsearch)

    '''
    
    config = {'host':host, 'port':port}

    try:
        es = Elasticsearch([config])

    except Exception as e:
        logging.error('Couldnt connect to ES server', exc_info=e)

    return es


def create_es_index(es_obj, settings, index_name):
    '''
    Create an ElasticSearch index

    Args:
        es_obj (elasticsearch.client.Elasticsearch)
        settings (dict)
        index_name (str)

    '''

    es_obj.indices.create(index=index_name, body=settings, ignore=400)

    logging.info('Index created successfully!')

    return

def load_es_index(es_obj, index_name, evidence_corpus):
    '''
    Loads records into an existing ElasticSearch index

    Args:
        es_obj (elasticsearch.client.Elasticsearch)
        index_name (str)
        evidence_corpus (list) - list of dicts containing data records

    '''

    for i, rec in enumerate(tqdm(evidence_corpus)):
    
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)

        except Exception as e:
            logging.error(f'Error loading doc with index {i}', exc_info=e)
    
    time.sleep(10)
    n_records = es_obj.count(index=index_name)['count']
    logging.info(f'Succesfully loaded {n_records} into {index_name}')

    return

def run_question_query(es_obj, index_name, question_text, n_results=5):
    '''

    '''

    # construct query
    query = {
            'query': {
                'query_string': {
                    'query': re.sub('[^A-Za-z0-9]+', ' ', question_text),
                    'default_field': 'document_text_clean'
                    }
                }
            }

    # execute query
    res = es_obj.search(index=index_name, body=query, size=n_results)

    return res