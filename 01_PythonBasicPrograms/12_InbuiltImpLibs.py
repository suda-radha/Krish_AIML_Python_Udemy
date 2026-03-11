#array lib
# math lib
# random lib
#os lib
# shutil lib for file operations
# json lib for data serialization - dumps() and load() methods
# csv lib to create/read csv file data
# time lib
# re lib for regex

import array
arr = array.array('i', [1,2,3,4])
print(arr)

import math
print(math.sqrt(25))
print(math.pi)

import random
print(random.randint(1,10))
print(random.choice(["apple", "banana", "mango", "grapes", "cherrry"]))

import os
print(os.getcwd())
#os.mkdir("testdir") # creates a folder testdir in current working dir
with open('file.txt', 'w') as f: # creates a new file in cwd
    pass


import shutil
shutil.copyfile("source.txt", "destination.txt")

# data serialization
import json
data = {'name': 'krish', 'age': 25}
json_str=json.dumps(data) # converts dict to json
print(json_str)
print(type(json_str))

json_dict = json.loads(json_str) # converts back json str to dict
print(json_dict)
print(type(json_dict))

import csv
with open("example.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['name', 'age'])
    writer.writerow(['krish', 25])
    writer.writerow(['rama', 24])

with open("example.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

from datetime import datetime,timedelta

print(datetime.now())
print(datetime.now()-timedelta(days=1))

import time
print(time.time())
time.sleep(2)
print(time.time())

import re
pattern = r'\d+' # digits pattern
text = "There are 124 apples in 8 baskets"
matched_text =re.search(pattern,text)
print(matched_text.group())