import json
import os
import sys
import re
import pickle
import logging
import gzip
import shutil
import urllib.request
from tqdm import tqdm
from collections import defaultdict

# --------------------------------------------- Data Prep Utils ----------------------------------------------------

def download_nq_train_data():
    '''
    Downloads and unzips the simplified verson of Google Natural Questions dataset.

    '''

    # create data directory in parent folder
    os.makedirs('../data', exist_ok=True)

     # create data directory for raw data in data directory
    os.makedirs('../data/raw_data', exist_ok=True)
    filepath = '../data/raw_data'

    # download data
    logging.info('Beginning to download Natural Questions Simplified Train data')

    url = 'https://00e9e64bac5fa14003581c8a55581c7f65d2b932543c109661-apidata.googleusercontent.com/download/storage/v1/b/natural_questions/o/v1.0-simplified%2Fsimplified-nq-train.jsonl.gz?qk=AD5uMEuhixdPI4RVW-Vp92Ctehkh0xgrPCBAuMKIT0EF0xHiLeqaOuQaFYTvLatFxlhRkylGfZEb0PpZwgda203f1EeK6IxrdoFKK3xDVZMsUqpF852yTnjj_aQmlM0NvLDMjoDBfza2N6oAnx3wdznwXZ-WJW0WfudIQT8tkBzo-Tt3mVvD5To5LjmdUI7a35GILzS4fmSix9tOGbaEVw4sSo3wSikMYNPfoBJU6Nfprr2VJjg_fuqBPfxqTciahIHBCyRKTuJVAxy40AxZGOHHBYbfmUcqfhTyS0IfbYy0pylvprptUZpivxRKkSn1dFxMXtIpV3sBh3D1Q04Q8I9Ygt2TX4kEIO6-DEtM3--odSgWKoc3sWRB7Vwv1nM5JxXdM049pI4tb3VFB7zch4_VfV0DggLtTYHnf5xsKOewtCxu8yvmhN0oshKo21ijgl8u6DvWi6HeEYT6ixvFbQRyB52hZmlxeegZh5dWvyeoGyQmxpGQLFWAGI7Lsst3232mTUBmE_H0Tj62YYGyc5kUQyqjkLMwiYqOK-dFdfAyW8yCKN2dg-XkjqoMZTg_7WaP0tnJ9cBrMD27n5Y3zSyPVjhkFk9gnSZOiJN14KUHpfNBVoprgkaXygyQWahv-69oa6qB2cjHEYmFgoPa24r_OPLPX9MtD1JHT5U8R4MsKveg5-RCPkd4h_Ot06YzoXsKsbJk_xNGwGX5ynjcrudEfrHpyACEfqN0aTGd1yZfhuBNS5ng0IEdjfvtQDzJyzwoKRjoqg9QQcUUAvHwMF11cWxUisQgIP6E0_lgHmsWITygeUeXi8pxR-XVzPFVOEO9NPCl4bZEPkTh9NZA1dBhjppmFv-LfQ&isca=1'
    filename = 'v1.0-simplified_simplified-nq-train.jsonl.gz'
    urllib.request.urlretrieve(url, filename=filepath+filename)

    logging.info('Download Complete')

    # unzip and save
    with gzip.open(filepath+filename, 'rb') as f:
        with open(filepath+filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f, f_out)
    
    logging.info(f'Data artifacts unzipped to {filepath}')

    return


def load_jsonl_file(filepath):
    '''
    Loads a jsonl file from disk given a path to the file

    Args:
        filepath (str) - path to jsonl file

    Returns:
        data (list) - Python list object representation of the jsonlines file

    '''

    logging.info('Loading Data')

    data = []
    with open(filepath, 'rb') as f:
        for i, line in enumerate(tqdm(f)):
            data.append(json.loads(line.decode('utf-8')))

    return data


def load_pkl_file(filepath):
    '''
    Loads a plk file from disk given a path to the file

    Args:
        filepath (str) - path to pkl file

    Returns:
        data (list) - python object

    '''

    logging.info('Loading Data')

    data = []
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def create_pkl_file(obj, filepath):
    '''
    Dumps an object from memory to disk

    Args:
        obj - object to pickle
        filepath (str) - location to save object

    '''

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    logging.info(f'Successfully pickled object: {filepath}')

    return








