# Insight-Wave

Insight-Wave is a research project focused on enhancing stock prediction accuracy using extensive online data. By examining the SET50 index and a representative sample of 19 stocks from 7 industries, the project leverages machine learning and deep learning techniques. This includes extracting topics from news and social media, and performing sentiment analysis using the GDELT database. Integrating these data sources aims to improve stock prediction accuracy and provide valuable insights to researchers and financial institutions.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgements](#acknowledgements)

## Introduction

Insight-Wave explores the use of extensive online data to enhance stock market prediction accuracy, focusing on the SET50 index. The project is divided into Topic Modeling and Stock Prediction. Topic Modeling extracts meaningful topics from online data, while Stock Prediction builds predictive models using this enriched dataset. By incorporating diverse data sources, the project aims to provide valuable insights and improve stock market prediction accuracy.


## Project Structure

The Insight-Wave project is organized into several key directories and files to ensure a clear and efficient workflow. Below is an overview of the main components:

```
Insight-Wave/
│
├── DataCollection/
│   └── src/
│       ├── Checking.ipynb         # Jupyter notebook for data checking
│       ├── Consolidate.py         # Script for data consolidation
│       ├── Crawling/              # Folder for web crawling scripts
│       └── Parsing/               # Folder for data parsing scripts
│      
├── StockPrediction/
│   ├── backtest/                  # Folder for backtesting results
│   ├── data/
│   │   ├── Fundamental+Technical Data/  # Folder for stocks’ fundamental and technical data
│   │   └── Online Data/           # Folder for input online data for stock prediction
│   ├── model/                     # Folder for models’ parameter tuning files
│   └── src/
│       ├── __pycache__/           
│       ├── command.py             # Main script to run stock prediction
│       └── prediction_function.py # Script for prediction functions
│
├── TopicModeling/
│   └── src/                       # Folder for topic modeling scripts
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project overview and instructions

```

## Installation

To set up the Insight-Wave project, follow these steps:

### Prerequisites

Ensure you have the following software installed on your system:

- Python 3.8 or higher
- Git

### Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/The-BigBox/Insight-Wave.git
cd Insight-Wave
```

### Setting Up Virtual Environments

It's recommended to use a virtual environment to manage dependencies. You can use ‘venv’ for this purpose:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

Install all required dependencies listed in the requirements.txt file:

```
pip install -r requirements.txt
```

## Usage

This section provides instructions on how to use the Insight-Wave project to perform stock predictions. 

### Data Collection

1. **Run Data Collection:**
 - The folder of Crawling contains the scripts and tools necessary for collecting raw news articles from various sources.
 - The folder of Parsing contains the Jupyter Notebooks and scripts used to parse the collected raw data into structured formats.
 - `Consolidate.py` script consolidates multiple text files containing news articles into a single text file. Each article should be separated by two newline characters.
 - `Checking.ipynb` notebook is used to perform various checks and verifications on the consolidated data. This includes checking for the total number of articles, identifying missing or corrupted articles, and ensuring that the data is in the correct format.

### Topic Modeling

1. **Prepare Data:**
   - Use the `tokenizer.py` script to preprocess your text data. This will tokenize the text, remove stopwords, and save the processed dataset. The customized stopwords list was provided in remove_from_stopword.txt and add_to_stopword.txt located in the src directory.

2. **Run Topic Modeling:**
- Set Up the Training Parameters
   - Define the number of topics, total passes, and other model parameters in the `Model.ipynb` notebook.
- Train the Model
   - Open `Model.ipynb` in Jupyter Notebook.
Execute the cells to train the LDA model on your preprocessed dataset.
The model will periodically save its state to the specified directory.

### Stock Prediction

1. **Prepare Data:**
   - Ensure all processed data files are placed in the appropriate directories under `StockPrediction/data`.
   - The data should include both fundamental and technical data, as well as online data.

2. **Run Stock Prediction:**
   - Navigate to the `StockPrediction` directory:
     ```
     cd StockPrediction
     ```
   - Execute the main script to start the stock prediction process:
     ```
     python src/command.py
     ```
   - The `command.py` script contains the logic to gather inputs and call the necessary functions from `prediction_function.py`. Make sure that the data is properly formatted and available in the specified directories.

### Analyzing Results

1. **Topic Modeling Results:**
   - The topic distribution for each document will be saved in a CSV file.
   - You can analyze this CSV file to understand the distribution of topics across your corpus.
   - You can view the identified topics and their associated words by running the relevant cells in the `Model.ipynb` notebook.

2. **Stock Predictions Result:**
   - Predictions and evaluations are saved in the `StockPrediction/model` folder.
   - Backtesting results are saved in the `StockPrediction/backtest` folder.
   - Navigate to these directories to review the prediction outputs and evaluation metrics.
   - Review the saved evaluation metrics such as Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and Directional Accuracy (DA).
   - Use these metrics to assess the accuracy and effectiveness of the prediction models.

### Customization

1. **Modify Parameters:**
   - To customize the models, modify the parameters in the scripts located in `StockPrediction/src` and `TopicModeling/src`.
   - Adjust hyperparameters, data sources, and other configurations as needed to improve model performance.

2. **Extend Functionality:**
   - You can add new data sources, models, or analysis methods by creating new scripts and integrating them into the existing workflow.
   - Ensure any new dependencies are added to the `requirements.txt` file and installed.

This completes the usage instructions for the Insight-Wave project. For any further customization or advanced usage, refer to the detailed comments and documentation within the scripts.

## License

This project is licensed under the Faculty of Information and Communication Technology, Mahidol University license.

## Acknowledgements

We would like to extend our gratitude to the following individuals and organizations:

- **Advisors:**
  - Dr. Petch Sajjacholapunt for his invaluable guidance and support throughout the research project.
  - Assoc. Prof. Dr. Suppawong Tuarob for his valuable insights and unwavering support.

- **Faculty of Information and Communication Technology, Mahidol University:**
  - For providing crucial hardware support and a conducive workspace, which were instrumental in achieving our project goals efficiently and effectively.

- **Contributors:**
  - Mr. Phumtham Akkarasiriwong ([@Oatttttttt](https://github.com/Oatttttttt))
  - Mr. Chayut Piyaosotsun ([@Chayut Piyaosotsun](https://github.com/ChayutPiyaosotsun))
  - Mr. Phurich Kongmont ([@Phurich Kongmont](https://github.com/PhurichKg))

- **Supporters:**
  - All individuals and organizations that contributed to the successful completion of this project.

Thank you for your support and contributions!

---



