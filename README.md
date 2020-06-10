# Question Answering Evaluation Framework



**TO-DO:** Fill this out



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
$ /usr/local/var/homebrew/linked/elasticsearch-full/bin
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
$ conda activate 
```



### Run Data Preparation Script

To streamline project setup, a script has been created to process the NQ train data and load it into an ElasticSearch index for testing. 

```shell
$ python3 prepare_data.py -R 0
```

After successfully running this script, access the `Example_Notebook.ipynb` for interacting with ElasticSearch