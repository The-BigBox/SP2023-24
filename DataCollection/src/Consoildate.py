
import os
import re
import pandas as pd
import chardet
from bs4 import BeautifulSoup
import csv


main = os.getcwd() + "/../../../"
findFile = main + "/DataCrawling/result"
output_path = main + "TopicModeling/data/ModelingDataset/V.2"

def merge(ag):

    # directories = [f for f in os.listdir(findFile) if os.path.isdir(os.path.join(findFile, f))]
    # print(directories)

    data_dict = []
    processed_article_ids = []
    df = pd.DataFrame(columns=["ID", "Agency", "Date", "Title", "Links", "Status"])
    df_new = pd.DataFrame(columns=["ID", "Agency", "Date", "Article"])

    df_data = []  
    df_new_data = []  

    print("--------------", ag)

    if ag == 1:
        directories = ['dailynews']
    elif ag == 2:
        directories = ['kaohoon']
    elif ag == 3:
        directories = ['prachachat']
    elif ag == 4:
        directories = ['thairath']
    elif ag == 5:
        directories = ['thansettakij']
    elif ag == 6:
        directories = ['thunhoon']

    for agent in directories:
        article_path = os.path.join(findFile, agent, "article")
        article_list = [article for article in os.listdir(article_path) if article != ".DS_Store" and not article.endswith('.zip')]  # Exclude .DS_Store files and .zip files
        article_list = sorted(article_list, key=int)
        print(f"Starting parsing the news agent as : {agent}")
        parse_total = []

        for article_id in article_list:
            
            data_path = os.path.join(article_path, article_id, "parsed.txt") # input path

            title = ""
            article = ""
            result = ""
            date = ""
            status = "False(NoPathExists)"  # Default to False status

            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()  # Read all lines at once
                    for i, line in enumerate(lines):  # Use enumerate to keep track of the line index
                        try:
                            if line.startswith('[::Title::]') and i + 1 < len(lines):
                                title = lines[i + 1].strip()
                            elif line.startswith('[::Article::]') and i + 1 < len(lines):
                                article = lines[i + 1].strip()
                            elif line.startswith('[::DateTime::]') and i + 1 < len(lines):
                                date = lines[i + 1].strip()
                        except StopIteration:
                            break  # Exit the loop if there are no more lines

                    if len(article) < 50:
                        continue
                        
                    status = "True"
                    
                    # Concatenate title and article
                    result = title + " " + article
                    data_dict.append(result)
                    parse_total.append(len(data_dict))
            if status == "False(NoPathExists)":
                continue
        
        
            if agent == "kaohoon":
                link = f'https://www.kaohoon.com/news/{article_id}'
            elif agent == 'thunhoon':
                link = f'https://thunhoon.com/article/{article_id}'
            elif agent == 'thansettakij':
                link = f'https://www.thansettakij.com/finance/stockmarket/{article_id}'
            elif agent == 'prachachat':
                link = f'https://www.prachachat.net/finance/news-{article_id}'
            elif agent == 'thairath':
                link = f'https://www.thairath.co.th/news/local/{article_id}'
            elif agent == 'dailynews':
                link = f'https://www.dailynews.co.th/news/{article_id}'


            # Append to list for DataFrame
            df_data.append({
                'ID': article_id,
                'Agency': agent,
                'Date': date,
                'Links': link, 
                'Title': title,
                'Status': status
            })

            df_new_data.append({
                'ID': article_id,
                'Agency': agent,
                'Date': date,
                'Article': result,
            })

            processed_article_ids.append(article_id)
    



    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(df_data)
    df_new = pd.DataFrame(df_new_data)

    # Save the data
    output_txt = os.path.join(output_path, "consoildate_data.txt")
    output_csv = os.path.join(output_path, "MasterDesctiption.csv")
    output_dataset = os.path.join(output_path, "dataset.csv")



    with open(output_txt, 'a', encoding='utf-8') as output_file:
        for entry in data_dict:
            output_file.write(entry + "\n\n")

    # Check if the file exists to decide whether to write headers
    header = not os.path.isfile(output_csv)

    # Append to the CSV file without writing the header if the file already exists
    df.to_csv(output_csv, mode='a', encoding="utf-8", index=False, header=header)

    # Repeat for the second CSV file
    header = not os.path.isfile(output_dataset)
    df_new.to_csv(output_dataset, mode='a', encoding="utf-8", index=False, header=header)


    print("Saved")
    print("-------------------Finish-------------------")



ag = int(input(" Enter "))
merge(ag)
