# How to search in a folder for files with *.dat in the name?

from glob import glob
import os

folder_path = "/path/to/your/folder"

target_files = glob(os.path.join(folder_path, "*.dat"))
print(dat_files)

# Output example: ['/path/to/your/folder/file1.dat', '/path/to/your/folder/file2.dat']
