import pandas as pd
import re
import json
import random
import os
import time

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamodel import LdaModel
import gensim

from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_words, thai_stopwords

# File paths
main_document = "/data1-6tb/sp2023stock/TopicModeling/data/ModelingDataset/consoildate_data.txt"  # Replace with your file path
PATH_SAVING_ASSETS = '/data1-6tb/sp2023stock/TopicModeling/model/ModelingAssets/'

# Read the main document
with open(main_document, 'r', encoding='utf-8') as file:
    content = file.read()

# Splitting the content to recreate the list of entries
pre_data_dict = content.split("\n\n")
pre_data_dict = [entry for entry in pre_data_dict if entry.strip()]
print(f"Total documents: {len(pre_data_dict)}")

sample_list = [100]

for sample_number in sample_list:
    directory = f'CorpusDict{sample_number}'
    path = os.path.join(PATH_SAVING_ASSETS, directory)

    # Calculate sample size
    total_documents = len(pre_data_dict)
    sample_size = int(total_documents * (sample_number / 100))

    # Randomly select documents
    print(f'Sampling {sample_number}% of the data')
    data_dict = random.sample(pre_data_dict, sample_size)
    print(f'Total length of the sampled data: {len(data_dict)}')

    # File paths for stopwords modifications
    remove_file_path = '/data1-6tb/sp2023stock/TopicModeling/src/remove_from_stopword.txt'
    add_file_path = '/data1-6tb/sp2023stock/TopicModeling/src/add_to_stopword.txt'

    # Read files for stopwords modifications
    with open(remove_file_path, 'r', encoding='utf-8') as file:
        remove_list = file.read().splitlines()
    with open(add_file_path, 'r', encoding='utf-8') as file:
        add_list = file.read().splitlines()

    # Stopwords and word set modifications
    stopwords = set(thai_stopwords())
    thaiwords = set(thai_words())

    # Adding and removing words from stopwords and thaiwords
    stopwords.update(add_list)
    thaiwords.update(remove_list)

    print(f'Stopwords: {len(stopwords)}, Thai words: {len(thaiwords)}')

    def preprocess(doc, stopwords):
        # Preprocessing steps
        doc = re.sub(r'http\S+', '', doc)  # Remove URLs
        tokens = word_tokenize(doc)  # Tokenization
        tokens = [token for token in tokens if (not re.search(r'[^a-zA-Z0-9ก-๙]', token)) and 
                  (not token.isdigit() or (token.isdigit() and len(token) == 4))]
        tokens = [token for token in tokens if not bool(re.match(r'^\d+\.\d+$', token))]
        tokens = [token for token in tokens if token not in stopwords]
        return tokens

    # Tokenize and preprocess the documents
    print("Tokenizing and preprocessing the documents...")
    dataset = [preprocess(doc, stopwords) for doc in data_dict]

    # Convert to Gensim Dictionary and Corpus
    dictionary = Dictionary(dataset)
    corpus = [dictionary.doc2bow(text) for text in dataset]

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save the Dictionary and Corpus
    dictionary.save(os.path.join(path, 'dictionary.gensim'))
    MmCorpus.serialize(os.path.join(path, 'corpus.mm'), corpus)

    print(f'Saving at {path}') 
    # Saving the preprocessed dataset
    json_file_path = os.path.join(PATH_SAVING_ASSETS, f'dataset_{sample_number}.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset, json_file, ensure_ascii=False, indent=2)

    print(f"Save Finished for sample {sample_number}")
    print(f"Number of documents in corpus: {len(corpus)}")

print("Processing complete.")

