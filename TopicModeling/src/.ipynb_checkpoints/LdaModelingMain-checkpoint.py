# Setting the JOBLIB_TEMP_FOLDER environment variable
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/data1-6tb/sp2023stock/TopicModeling/data/temp'

# Importing necessary libraries
import gensim
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamodel import LdaModel
from pythainlp import word_tokenize, Tokenizer
from pythainlp.util import normalize
import pythainlp.tokenize 
from pythainlp.corpus.common import thai_words, thai_stopwords

# Load the dictionary
dictionary = Dictionary.load('/data1-6tb/sp2023stock/TopicModeling/data/ModelingAssets/CorpusDict/dictionary.gensim')

# Load the corpus
corpus = MmCorpus('/data1-6tb/sp2023stock/TopicModeling/data/ModelingAssets/CorpusDict/corpus.mm')

print("Dictionary and Corpus loaded successfully.")

print(len(corpus))

# Initialize the model
num_topics = 200
lda = gensim.models.ldamulticore.LdaMulticore(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    workers= 16
)

# Number of total passes and intervals to save
total_passes = 1
save_interval = 100
save_path = "/data1-6tb/sp2023stock/TopicModeling/data/ModeingState"

# Train and save periodically
for pass_num in range(total_passes):
    lda.update(corpus)
    if (pass_num + 1) % save_interval == 0:
        filename = f"{save_path}lda_model_pass_{pass_num + 1}"
        lda.save(filename)
        print(f"Saved model at pass number {pass_num + 1} to {filename}")

# Display topics
print("Discovered Topics:")
for i in range(num_topics):
    words = lda.show_topic(i, 25)
    print(f"Topic {i}:", [word[0] for word in words])


import pandas as pd

# Initialize a list to store the topics data
topics_data = []

# Retrieve the topics and their words
print("Discovered Topics:")
for i in range(num_topics):
    words = lda.show_topic(i, 25)
    topic_words = [word[0] for word in words]
    topics_data.append(topic_words)
    print(f"Topic {i}:", topic_words)

# Create a DataFrame from the topics data
df_topics = pd.DataFrame(topics_data)
df_topics.index = [f"Topic {i}" for i in range(num_topics)]
df_topics.columns = [f"Word {i+1}" for i in range(25)]

# Save the DataFrame to a CSV file
csv_file_path = '/data1-6tb/sp2023stock/TopicModeling/result/topics.csv'
df_topics.to_csv(csv_file_path, index=True)

print(f"Topics saved to {csv_file_path}")
