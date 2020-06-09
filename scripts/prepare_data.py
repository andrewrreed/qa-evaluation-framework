import logging
import argparse
import sys
import os

module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import data_utils

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(levelname)s-%(message)s')


# parse cli args
parser = argparse.ArgumentParser()
parser.add_argument('-RO', '--retriever-eval-only', help='1 if running retriever pipeline only, else 0')
args = parser.parse_args()

if not args.retriever_eval_only:
    raise Exception('Must specifify -RO argument (of 1 or 0) indicating running pipeline for retriever-only or for full system.')
else:
    retriever_eval_only = bool(args.retriever_eval_only)

print(retriever_eval_only)

# run data prep pipeline
if not os.path.exists(os.path.join(module_path,'data/stage_data')):
    data_utils.data_prep_pipeline(retriever_eval_only)

# run data compilation pipeline
if not os.path.exists(os.path.join(module_path,'data/eval_data')):
    data_utils.compile_data_pipeline(retriever_eval_only)