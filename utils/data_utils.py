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


def filter_nq_train_data(raw_data, retriever_eval_only=True):
    '''
    This function takes the full corpus of NQ training data and filters examples that
    are not relevant for proper retriever evaluation, including: 

        a.) records that do not have at least one short answer are discarded and 
        b.) records that have more than one short answer are truncated to only use the first short answer.
    
    These filters are in line with standard retriever evaluation techniques as well as
    Google's suggested reference implementation:
    
    https://github.com/google-research/language/blob/master/language/question_answering/
    decatt_docreader/preprocessing/create_nq_short_pipeline_examples.py
    
    Args:
        raw_data (list) - python object representation of the raw jsonl file
        retriever_eval_only (bool) - if False, include short answer AND no answer
        
    Returns:
        filtered_data (list) - a refined version of the raw jsonl file
    
    '''

    logging.info('Filtering Data')
    
    multi_count = 0 
    filtered_data = []
    
    for i, rec in enumerate(tqdm(raw_data)):
        
        # ignore questions that dont have at least one short answer
        if len(rec['annotations'][0]['short_answers']) == 0 and retriever_eval_only==True:
            continue
        
        # if an annotation contains multiple short answers, keep only the first
        if len(rec['annotations'][0]['short_answers']) > 1:
            
            multi_count += 1
            
            # extract first dict and keep as one-element list
            temp = []
            short_ans = rec['annotations'][0]['short_answers'][0]
            temp.append(short_ans)
            
            # overwrite
            new_rec = rec.copy()
            new_rec['annotations'][0]['short_answers'] = temp
            
            filtered_data.append(new_rec)
        
        else:
            filtered_data.append(rec)
            
            
    logging.info(f'{len(raw_data)-len(filtered_data)} records (out of {len(raw_data)}) did not have at least one short answer and were dropped.')
    logging.info(f'{multi_count} questions had multiple short answers that were effected by truncation.')
            
    return filtered_data


def get_short_answer_from_span(example):
    '''
    Use the short answer span from a NQ json record to retreive
    and return the corresponding short answer text.
    
    Args:
        example - a jsonl record from NQ simplified dataset
        
    Returns:
        short_answer (string) - the string representation of text in the short answer span
    
    '''
    
    short_answer_span = example['annotations'][0]['short_answers'][0]
    
    short_answer = " ".join(example['document_text'].split(" ")\
                            [short_answer_span['start_token']:short_answer_span['end_token']])
    
    return short_answer


def clean_document_text(text):
    '''
    This function applies a regular expression to an input text string to remove
    any characters wrapped in <> with the goal of stripping HTML tags from a string.
    
    Args:
        text (string)
        
    Returns:
        text (string) - cleaned text
    
    '''
    
    cleaner = re.compile('<.*?>')
    
    return re.sub(cleaner, '', text)


def extract_wiki_title(document_url):
    '''
    This function applies a regular expression to an input wikipedia article URL
    to extract and return the article title.
    
    Args:
        document_url (string)
        
    Returns:
        title (string) - article title
    '''
    
    pattern = 'title=(.*?)&amp'
    
    try:
        title = re.search(pattern, document_url).group(1)
    except AttributeError:
        title = 'No Title Found'
        
    return title
    
    
def extract_data(data):
    '''
    This function loops through a list of NQ simplified records and extracts only the data items
    needed for retriever evaluation including:
        - example_id
        - document_title (extracted from document_url using extract_wiki_title())
        - document_url
        - question_text
        - short_answer (converted to text using get_short_answer_from_span())
        - document_text_clean (stripped of remaining HTML tags using clean_document_text())
    
    Args:
        data (list) - a list of filtered jsonl records from NQ simplified dataset
        
    Returns:
        extracted_data (list) - a list of cleaned jsonl records
    
    '''
    
    logging.info('Extracting Data')

    extracted_data = []
    
    for i, rec in enumerate(tqdm(data)):
        
        example_id = rec['example_id']
        document_url = rec['document_url']
        question_text = rec['question_text']
        short_answer = get_short_answer_from_span(rec)
        document_text_clean = clean_document_text(rec['document_text'])
        document_title = extract_wiki_title(rec['document_url'])
        
        # to ensure our dataset is completely solveable this logic weeds out erroneous labels
        # ex. 'Mickey Hart </Li> <Li> Bill Kreutzmann </Li> <Li> John Mayer </Li> was selected as long AND short answer
        # when really each of these should have been their own short answers
        if short_answer not in document_text_clean:
            continue
        
        new_rec = {'example_id': example_id,
                   'document_title': document_title,
                   'document_url': document_url,
                   'question_text': question_text,
                   'short_answer': short_answer,
                   'document_text_clean': document_text_clean}
        
        extracted_data.append(new_rec)
        
    logging.info(f'{len(extracted_data)} of the {len(data)} records are complete and solvable.')
    
    return extracted_data
    

def drop_longer_answers(data):
    '''
    This function loops through a list of NQ simplified records and drops any records where the short answer
    contains more than 5 tokens. 
    
    Answers with many tokens often resemble extractive snippets rather than canonical answers, so we discard
    answers with more than 5 tokens: https://arxiv.org/pdf/1906.00300.pdf
    
    Args:
        data (list) - a list of cleaned jsonl records from NQ simplified dataset
        
    Returns:
        extracted_data (list) - a list of cleaned jsonl records
    
    '''
    
    logging.info('Dropping Long Answers')

    slim_data = []
    
    for i, rec in enumerate(tqdm(data)):
        
        if len(rec['short_answer'].split(' ')) <= 5:
            slim_data.append(rec)
            
    logging.info(f'{len(data) - len(slim_data)} records were "long" short-answers and were dropped.')
    logging.info(f'{len(slim_data)} records remain.')
            
    return slim_data


def compile_evidence_corpus(extracted_data):
    '''
    This function compiles all unique wikipedia documents into a dictionary
    
    Args:
        extracted_data (list) 
        
    Returns:
        evidence_docs (dict)
    
    '''

    logging.info('Compiling Evidence Docs')
    
    unique_titles = []
    evidence_docs = []
    
    for i, rec in enumerate(tqdm(extracted_data)):
        
        if rec['document_title'] not in unique_titles:
            
            unique_titles.append(rec['document_title'])
            
            fields = {'document_title': rec['document_title'],
                      'document_url': rec['document_url'],
                      'document_text_clean': rec['document_text_clean']}
            
            evidence_docs.append(fields)
                
            
    print(f'Of the {len(extracted_data)} records, there are {len(evidence_docs)} unique Wikipedia articles.')
        
    return evidence_docs


def compile_qa_records(extracted_data):
    '''
    This function loops through the extracted_clean_data list and removes the document_text_clean field
    from each record
    
    Args:
        extracted_data (list) 
        
    Returns:
        slim_data (list)
    '''
    
    logging.info('Compiling QA Records')

    slim_data = []
    
    for i, rec in enumerate(tqdm(extracted_data)):
        
        new_rec = {k:v for k,v in rec.items() if k != 'document_text_clean'}
        slim_data.append(new_rec)
        
    return slim_data


# --------------------------------------------- Pipelines ----------------------------------------------------

def prep_data_pipeline(retriever_eval_only=True):
    '''
    Data Preparation Utility Pipeline

    Calling this function executes a data processing pipeline that:
        1. Downloads Natural Questions simplified training dataset (if not already downloaded)
        2. Filters the examples to only those relevant to retriever evaluation (has short_answer, resolves multiple answers)
        3. Cleans, parses, and extracts relevant data fields 
        4. Saves the prepared data to a local directory

    '''

    logging.info('Data Preparation Pipeline Started')

    #download_nq_train_data()
    data = load_jsonl_file(filepath='../data/raw_data/v1.0-simplified_simplified-nq-train.jsonl') ## TO-DO: Make this implicit!
    data = filter_nq_train_data(data, retriever_eval_only)
    data = extract_data(data)
    data = drop_longer_answers(data)

    # create data directory in parent folder
    os.makedirs('../data/stage_data', exist_ok=True)


    if retriever_eval_only:
        outfile = '../data/stage_data/extracted_clean_data.pkl' ## TO-DO: Make this implicit!   
    else:
        outfile = '../data/stage_data/extracted_clean_data_fullsys.pkl' ## TO-DO: Make this implicit!   

    with open(outfile, 'wb') as f:  ## TO-DO: Make this implicit!
        pickle.dump(data, f)


    logging.info('Data Preparation Pipeline Finished')

    return


def compile_data_pipeline(retriever_eval_only=True):
    '''
    Data Compilation Utility Pipeline

    Calling this function executes a data pipeline that:
        1. Loads pre-cleaned data from staging
        2. Deduplicates Wikipedia artilces and finalizes them for loading into ElasticSearch
        3. Creates q/a records to be used for evaluation
        4. Saves those data artifacts to eval_data directory

    '''

    logging.info('Data Compilation Pipeline Started')

    # run pipeline
    data = load_pkl_file(filepath='../data/stage_data/extracted_clean_data.pkl') ## TO-DO: Make this implicit!
    evidence_corpus = compile_evidence_corpus(data)
    # TO-DO: insert function here for different chunking methods
    qa_records = compile_qa_records(data)

    # create data directory in parent folder
    os.makedirs('../data/eval_data', exist_ok=True)

    with open('../data/eval_data/evidence_corpus.pkl', 'wb') as f:
        pickle.dump(evidence_corpus, f)

    with open('../data/eval_data/qa_records.pkl', 'wb') as f:
        pickle.dump(qa_records, f)

    logging.info('Data Compilation Pipeline Finished')

    return







