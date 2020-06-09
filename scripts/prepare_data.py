import logging
import sys
import os

module_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import data_utils

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s-%(levelname)s-%(message)s')



# run data prep pipeline
if not os.path.exists(os.path.join(module_path,'data/stage_data')):
    data_utils.data_prep_pipeline()

# run data compilation pipeline
if not os.path.exists(os.path.join(module_path,'data/eval_data')):
    data_utils.compile_data_pipeline()