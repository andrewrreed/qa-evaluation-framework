import logging
import argparse
import sys
import os

from routines import DataPreprocessingRoutine, DataCompilationRoutine

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(levelname)s-%(message)s')


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


# instantiate routines
raw_data_path = 'data/raw_data/TESTING_v1.0-simplified_simplified-nq-train.jsonl'
dpr = DataPreprocessingRoutine(raw_data_path=raw_data_path,
                               retriever_eval_only=retriever_eval_only)

dcr = DataCompilationRoutine(retriever_eval_only=retriever_eval_only)

# execute routines
dpr.run()
dcr.run()
