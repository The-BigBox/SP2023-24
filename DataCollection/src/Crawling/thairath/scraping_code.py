import os
import csv
import time
import random
import signal
import requests
from bs4 import BeautifulSoup
from requests.exceptions import ChunkedEncodingError

# Set the article IDs range
start_id = 2684420 
end_id = 2753686

# Set the delay time range (in seconds)
min_delay = 55
max_delay = 333

# Set the article folder name
article_folder = 'C:/Users/SPJ-2023/Documents/sp2023/Dataset/New_crawling/thairath'

def print_summary(page_count, last_visited_url, article_id, start_id):
    print(f'Saved {page_count} pages. Last visited URL: {last_visited_url}. Total pages visited: {article_id - start_id + 1}.')

def interrupt_handler(sig, frame):
    print('\nInterrupt signal received. Quitting...')
    print_summary(page_count, last_visited_url, article_id, start_id)
    exit()

# Register the interrupt handler
signal.signal(signal.SIGINT, interrupt_handler)

# Check if the article folder exists, if not create it
if not os.path.exists(article_folder):
    os.makedirs(article_folder)

# Check if there's a progress file, if not start from the beginning
if os.path.exists('progress.txt'):
    with open('progress.txt', 'r') as f:
        last_visited_id = int(f.readline().strip())
        current_id = last_visited_id + 1
else:
    current_id = start_id
    last_visited_id = current_id - 1

# Initialize the page count and the last URL visited
page_count = 0
last_visited_url = ''

for article_id in range(current_id, end_id + 1):
    url = f'https://www.thairath.co.th/news/local/{article_id}'
    delay = random.randint(min_delay, max_delay) / 100
    print(f"Crawling {url} in {delay} seconds...")
    time.sleep(delay)
    
    try:
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
        html_pretty = soup.prettify()
        last_visited_url = url

        with open('progress.txt', 'w') as f:
            f.write(str(article_id) + '\n')

        if '<img alt="ขออภัยเราไม่พบข่าวจากคำที่คุณค้นหา แต่เรามีข่าวที่น่าสนใจอีกมากมาย ลองเลือกดูด้านล่างนี้ได้เลย" src="/static/img/error/error404.png"/>' not in html_pretty:
            folder_name = os.path.join(article_folder, str(article_id))
            os.makedirs(folder_name, exist_ok=True)
            file_name = os.path.join(folder_name, 'index.txt')

            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(html_pretty)
            with open('numbers.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                validation = 'Valid'
                writer.writerow([article_id, validation])
            page_count += 1
        else: 
            with open('numbers.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                validation = 'Not Valid'
                writer.writerow([article_id, validation])
            page_count += 1


    except ChunkedEncodingError:
        print(f"ChunkedEncodingError encountered for URL: {url}. Skipping...")

    except requests.exceptions.ConnectionError:
        print(f"ConnectionError encountered for {url}. Skipping to the next article...")

print_summary(page_count, last_visited_url, article_id, start_id)
