# Question Answering Evaluation Framework

In a open domain question answering system, a two-stage approach is often employed to a.) search a large corpus and retrieve candidate documents based on an input query and b.) apply a reading comprehension algorithm to process the candidate documents and extract a specific answer that best satisfies the input query. 

This repo provides a framework for evaluating an end-to-end question answering system on the [Natural Questions](https://ai.google.com/research/NaturalQuestions/) (NQ) dataset. The framework consists of 3 parts:

1. Automated data preprocessing utilities for the NQ dataset
2. Automated ElasticSearch setup to enable candidate document retrieval capability
3. A pre-trained BERT reader and evaluation tools for machine reading comprehension

### The Dataset

Google's [Natural Questions](https://ai.google.com/research/NaturalQuestions/) corpus - a question answering dataset - consists of real, anonymized, aggregated queries issued to the Google search engine that are paired with high quality, manual annotations. The nature by which question/answer examples are created presents a unique challenge compared to previous question answering datasets making solutions to this task much more representative of true open-domain question answering systems.

### Document Retriever

ElasticSearch - a distributed, open source search engine built on Apache Lucene - is used in the framework for retrieving candidate documents. ElasticSearch utilizes the BM25 algorithm for information retrieval based on exact keyword matches and offers an extensive and easy to used API for search setup and interaction.

### Document Reader

Transformer based reading comprehension approaches have started to outperform humans on some key NLP benchmarks including question answering. Specifically, we utilize versions of BERT pre-trained on SQuAD2.0 to understand how well that find-tuned model performs on the more difficult NQ dataset.

## Getting Started

After cloning this repository, use the following instructions to setup project dependencies

### Download Dataset

- Download Natural Question's Simplified Train Set from [here](https://ai.google.com/research/NaturalQuestions/download)
- Extract the gzip file contents to the `data/raw_data` project folder



### Download, Install, Launch ElasticSearch

On MacOS:

```shell
# download
$ brew tap elastic/tap

# install
$ brew install elastic/tap/elasticsearch-full

# launch
$ elasticsearch
```



### Environment Setup

Ensure that the [Anaconda](https://www.anaconda.com/distribution/) distribution of Python is installed globally on your machine.


#### Create a Virtual Environment

To replicate the virtual environment, open an Anaconda shell and run:

```shell
# Create a replicate virtual environment
$ conda env create -f environment.yml
```

#### Activate the Environment

To activate the newly created Anaconda environment, run:

```shell
# Activate the virtual environment
$ conda activate qa-retriever-eval
```



### Run Data Preparation Script

To streamline project setup, a script has been created to process the NQ train data and load it into an ElasticSearch index for testing. 

```shell
$ python3 prepare_data.py -R 0
```

After successfully running this script, access the `Example_Notebook.ipynb` for interacting with the Document Retriever and Document Reader