import logging
import argparse
import sys
import os
from elasticsearch import Elasticsearch

from routines import DataPreprocessingRoutine, DataCompilationRoutine
from utils.elastic_utils import connect_es, create_es_index, load_es_index
from utils.data_utils import load_pkl_file

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(levelname)s-%(message)s')

module_path = os.path.dirname(os.path.abspath(__file__))


# parse cli args
parser = argparse.ArgumentParser()
parser.add_argument('-R', '--retriever-eval-only', help='1 if running pipeline for retriever only, else 0')
args = parser.parse_args()

if not args.retriever_eval_only:
    raise Exception('Must specifify -RO argument (of 1 or 0) indicating running pipeline for retriever-only or for full system.')
else:
    retriever_eval_only = bool(int(args.retriever_eval_only))


# ensure data has been manually downloaded
if not os.path.exists('data/raw_data/v1.0-simplified_simplified-nq-train.jsonl'):
    raise Exception('You must manually download the NQ Simplified trainset per the README.md')


# ---------------------------------------------- Data Prep ----------------------------------------------
# instantiate routines
raw_data_path = 'data/raw_data/TESTING_v1.0-simplified_simplified-nq-train.jsonl'
dpr = DataPreprocessingRoutine(raw_data_path=raw_data_path,
                               retriever_eval_only=retriever_eval_only)

dcr = DataCompilationRoutine(retriever_eval_only=retriever_eval_only)

# execute routines
dpr.run()
dcr.run()


#---------------------------------------------- Elastic Prep ----------------------------------------------

# instantiate ElasticSearch instance
es = connect_es()

# create an index
index_name = 'demo_index'
settings = {
        "mappings": {
            "dynamic": "strict",        
            "properties": {
                "document_title": {"type": "text"},
                "document_url": {"type": "text"},
                "document_text_clean": {"type": "text"}
                }
            }
        }

create_es_index(es_obj=es, settings=settings, index_name=index_name)

# load NQ articles into index
ext = "" if retriever_eval_only else "_fullsys"
ec_filepath = module_path+f'/data/eval_data/evidence_corpus{ext}.pkl'
evidence_corpus = load_pkl_file(ec_filepath)

load_es_index(es_obj=es, 
              index_name=index_name,
              evidence_corpus=evidence_corpus)

