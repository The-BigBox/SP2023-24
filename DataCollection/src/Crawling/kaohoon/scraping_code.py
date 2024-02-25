import os
import csv
import time
import random
import signal
from bs4 import BeautifulSoup
import requests

# Constants
start_id = 1
end_id = 646621 #646621
min_delay = 5
max_delay = 333
article_folder = 'C:/Users/SPJ-2023/Documents/sp2023/Dataset/New_crawling/kaohoon'

def print_summary(page_count, last_visited_url, article_id, start_id):
    print(f'Saved {page_count} pages. Last visited URL: {last_visited_url}. Total pages visited: {article_id - start_id + 1}.')

def interrupt_handler(sig, frame):
    print('\nInterrupt signal received. Quitting...')
    print_summary(page_count, last_visited_url, article_id, start_id)
    exit()

signal.signal(signal.SIGINT, interrupt_handler)


if not os.path.exists(article_folder):
    os.makedirs(article_folder)

if os.path.exists('progress.txt'):
    with open('progress.txt', 'r') as f:
        last_visited_id = int(f.readline().strip())
        current_id = last_visited_id + 1
else:
    current_id = start_id
    last_visited_id = current_id - 1

page_count = 0
last_visited_url = ''

for article_id in range(current_id, end_id + 1):
    url = f'https://www.kaohoon.com/breakingnews/{article_id}'
    delay = random.randint(min_delay, max_delay)/100
    print(f"Crawling {url} in {delay} seconds...")
    time.sleep(delay)

    response = requests.get(url)
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')
    html_pretty = soup.prettify()
    last_visited_url = url

    with open('progress.txt', 'w') as f:
        f.write(str(article_id) + '\n')

    if '<meta content="Page not found • ข่าวหุ้นธุรกิจออนไลน์" property="og:title"/>' not in html_pretty:
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

print_summary(page_count, last_visited_url, article_id, start_id)
