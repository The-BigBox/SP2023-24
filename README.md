# Insigh Wave 

## Installation


### How to Use the topic modeling
Utilizing Latent Dirichlet Allocation (LDA) to model topics in a corpus of news articles. Follow these steps to set up and use the project:

#### Preprocessing

- Use the tokenizer.py script to preprocess your text data. This will tokenize the text, remove stopwords, and save the processed dataset. The customized stopwords list was provided in remove_from_stopword.txt and add_to_stopword.txt located in the src directory.


####  Training the LDA Model
1. Set Up the Training Parameters
- Define the number of topics, total passes, and other model parameters in the Model.ipynb notebook.
2. Train the Model
  - Open Model.ipynb in Jupyter Notebook.
  - Execute the cells to train the LDA model on your preprocessed dataset.
  - The model will periodically save its state to the specified directory.
#### Evaluation
1. Evaluate the Model:
- Use the Evaluation.ipynb notebook to evaluate the coherence and perplexity of the trained LDA model.
- Open Evaluation.ipynb in Jupyter Notebook.
- Execute the cells to calculate coherence scores and visualize the results.
#### Output and Analysis
1. Topic Distribution

- The topic distribution for each document will be saved in a CSV file.
- You can analyze this CSV file to understand the distribution of topics across your corpus.
2. Word representative
  
- You can view the identified topics and their associated words by running the relevant cells in the Model.ipynb notebook.
