import os
print(os.getcwd())
#os.mkdir("testdir") # creates a folder testdir in current working dir
with open('file2.txt', 'w') as f: # creates a new file in cwd
    pass

# create new dir
#os.mkdir("package")
print(f"new dire package created")

# list all files and dirs
items= os.listdir('..') # . is for current dir
print(items)
for item in items:
    print(item)

# joining paths
dir_name= '../package'
file_name="file1.txt"
full_path=os.path.join(dir_name, file_name) # relative path
print(full_path)
full_path=os.path.join(os.getcwd(),dir_name, file_name) # absolute path
print(full_path)

# if file exist print it exists else print it does not exista nd create one
if os.path.exists("example1.txt"):
    print(f"file example1.txt exists")
else:
    print("file example1.txt does not exist, creating new file")
    with open("example1.txt", "w") as file:
        file.write("Hello, this is a sample text file.\n")
        file.write("This file was created using Python.")

