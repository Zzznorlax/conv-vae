import os

file_path = "/path/to/parent/folder/file.txt"
parent_dir_name = file_path.split(os.sep)[-2]

print(parent_dir_name)  # prints "parent"
