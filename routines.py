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

from utils.data_utils import load_jsonl_file, create_pkl_file, load_pkl_file

module_path = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------- Pipelines ----------------------------------------------------

class DataPreprocessingRoutine:
    '''
    Data Preparation Routine

    This class holds utilities that execute a data processing routine that:
        1. Downloads Natural Questions simplified training dataset (if not already downloaded)
        2. Filters the examples to only those relevant to retriever evaluation (has short_answer, resolves multiple answers)
        3. Cleans, parses, and extracts relevant data fields 
        4. Saves the prepared data to a local directory

    Args:
        retriever_eval_only (bool) - if False, pipeline includes short answer AND no answer
        raw_data_path (str) - path to unzipped simplified nq jsonl file

    '''

    def __init__(self, raw_data_path, retriever_eval_only=True):

        self.mode = retriever_eval_only
        self.raw_data_path = raw_data_path

    def run(self):

        logging.info('Data Processing Routine Started')

        # check if file already exits
        ext = "" if self.mode else "_fullsys"
        outfile = module_path+f'/data/stage_data/extracted_clean_data{ext}.pkl' ## TO-DO: Make this implicit!   
        
        if os.path.exists(outfile):
            raise Exception('This file has already been created. Please delete it if you wish to recreate.:', outfile)
            
        # run pipeline
        self.load_data()
        self.filter_nq_train_data()
        self.extract_data()
        self.drop_longer_answers()

        # save data 
        os.makedirs(module_path+'/data/stage_data', exist_ok=True)
        self.save_data(outfile)

        logging.info('Data Processing Routine Finished')

        return


    def load_data(self):
        '''
        Loads raw, zipped jsonl data from disk

        '''
        self.data = load_jsonl_file(filepath=self.raw_data_path)

        return

    def filter_nq_train_data(self):
        '''
        This method takes the full corpus of NQ training data and filters examples that
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
        
        for i, rec in enumerate(tqdm(self.data)):
            
            # ignore questions that dont have at least one short answer
            if len(rec['annotations'][0]['short_answers']) == 0 and self.mode==True:
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
                
                
        logging.info(f'{len(self.data)-len(filtered_data)} records (out of {len(self.data)}) did not have at least one short answer and were dropped.')
        logging.info(f'{multi_count} questions had multiple short answers that were effected by truncation.')
        
        # overwrite data attribute
        self.data = filtered_data

        return


    def extract_data(self):
        '''
        This method loops through a list of NQ simplified records and extracts only the data items
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
        
        for i, rec in enumerate(tqdm(self.data)):
            
            try:
                example_id = rec['example_id']
                document_url = rec['document_url']
                question_text = rec['question_text']
                short_answer = self.get_short_answer_from_span(rec)
                document_text_clean = self.clean_document_text(rec['document_text'])
                document_title = self.extract_wiki_title(rec['document_url'])
                
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

            except Exception as e:
                logging.info(str(e))
                continue

            
        logging.info(f'{len(extracted_data)} of the {len(self.data)} records are complete and solvable.')
        
        # overwrite data attribute
        self.data = extracted_data

        return

    def drop_longer_answers(self):
        '''
        This method loops through a list of NQ simplified records and drops any records where the short answer
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
        
        for i, rec in enumerate(tqdm(self.data)):
            
            if len(rec['short_answer'].split(' ')) <= 5:
                slim_data.append(rec)
                
        logging.info(f'{len(self.data) - len(slim_data)} records were "long" short-answers and were dropped.')
        logging.info(f'{len(slim_data)} records remain.')
                
        # overwrite data attribute
        self.data = slim_data

        return

    def save_data(self, outfile):
        '''
        Saves the data attribute to a pickle local file

        '''
        create_pkl_file(self.data, outfile)

        return

    @staticmethod
    def get_short_answer_from_span(example):
        '''
        Use the short answer span from a NQ json record to retreive
        and return the corresponding short answer text.
        
        Args:
            example - a jsonl record from NQ simplified dataset
            
        Returns:
            short_answer (string) - the string representation of text in the short answer span
        
        '''
        
        sa_field = example['annotations'][0]['short_answers']

        if len(sa_field) >= 1:
            short_answer_span = sa_field[0]
            
            short_answer = " ".join(example['document_text'].split(" ")\
                                    [short_answer_span['start_token']:short_answer_span['end_token']])
        else:
            short_answer = ''
            
        return short_answer

    @staticmethod
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

    @staticmethod
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



class DataCompilationRoutine:
    '''
    Data Compilation Utility Pipeline

    This class holds utilties to execute a data routine that:
        1. Loads pre-cleaned data from staging
        2. Deduplicates Wikipedia artilces and finalizes them for loading into ElasticSearch
        3. Creates q/a records to be used for evaluation
        4. Saves those data artifacts to eval_data directory

    Args:
        retriever_eval_only (bool) - if False, pipeline includes short answer AND no answer

    '''

    def __init__(self, clean_data_path=None, retriever_eval_only=True):

        self.mode = retriever_eval_only

        # set clean data path 
        ext = "" if self.mode else "_fullsys"
        self.clean_data_path = clean_data_path if clean_data_path else module_path+f'/data/stage_data/extracted_clean_data{ext}.pkl'

    def run(self):

        logging.info('Data Compilation Routine Started')

        # check if exists
        ext = "" if self.mode else "_fullsys"
        outfile_ec = module_path+f'/data/eval_data/evidence_corpus{ext}.pkl'
        outfile_rec = module_path+f'/data/eval_data/qa_records{ext}.pkl'


        if os.path.exists(outfile_ec) or os.path.exists(outfile_ec):
            raise Exception('These files have already been created. Please delete both to re-run.')
        
        self.load_data()
        self.compile_evidence_corpus()
        self.compile_qa_records()

        # save data
        os.makedirs(module_path+'/data/eval_data', exist_ok=True)
        self.save_data(self.evidence_corpus, outfile_ec)
        self.save_data(self.qa_records, outfile_rec)

        logging.info('Data Compilation Routine Finished')


    def load_data(self):
        '''
        Loads clean, extracted pickle file from disk

        '''
        self.data = load_pkl_file(filepath=self.clean_data_path)

        return

    def compile_evidence_corpus(self):
        '''
        This method compiles all unique wikipedia documents into a dictionary
        
        Args:
            extracted_data (list) 
            
        Returns:
            evidence_docs (dict)
        
        '''

        logging.info('Compiling Evidence Docs')
        
        unique_titles = []
        evidence_docs = []
        
        for i, rec in enumerate(tqdm(self.data)):
            
            if rec['document_title'] not in unique_titles:
                
                unique_titles.append(rec['document_title'])
                
                fields = {'document_title': rec['document_title'],
                         'document_url': rec['document_url'],
                         'document_text_clean': rec['document_text_clean']}
                
                evidence_docs.append(fields)
                    
                
        logging.info(f'Of the {len(self.data)} records, there are {len(evidence_docs)} unique Wikipedia articles.')
        
        self.evidence_corpus = evidence_docs

        return 

    def compile_qa_records(self):
        '''
        This method loops through the extracted_clean_data list and removes the document_text_clean field
        from each record
        
        Args:
            extracted_data (list) 
            
        Returns:
            slim_data (list)
        '''
        
        logging.info('Compiling QA Records')

        qa_records = []
        
        for i, rec in enumerate(tqdm(self.data)):
            
            new_rec = {k:v for k,v in rec.items() if k != 'document_text_clean'}
            qa_records.append(new_rec)

        self.qa_records = qa_records
            
        return 

    @staticmethod
    def save_data(obj, outfile):
        '''
        Saves the obj  to a pickle local file

        '''
        create_pkl_file(obj, outfile)

        return