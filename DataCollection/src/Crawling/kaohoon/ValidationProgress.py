import os
import csv

path = 'D:/sp2023-stock-crawler/News Crawling/kaohoon/article'

# Use os.listdir to get the names of files and directories in the path
names = os.listdir(path)

# Filter the list to include only directories and sort it
folder_names = sorted([name for name in names if os.path.isdir(os.path.join(path, name))])

if os.path.exists('progress.txt'):
    with open('progress.txt', 'r') as f:
        last_visited_id = int(f.readline().strip())
        current_id = last_visited_id + 1

# Open the CSV file in write mode
with open('numbers.csv', 'w', newline='') as f:
    # Create a CSV writer
    writer = csv.writer(f)

    for i in range(current_id):
        validation = 'Valid' if str(i) in folder_names else 'Not Valid'
        writer.writerow([i, validation])
