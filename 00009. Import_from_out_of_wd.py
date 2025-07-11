import sys 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from file_name import function_you_want




###### relative path from your current working directory (cwd) to the Python script i want:
import os
script_path = os.path.abspath(__file__) # absolyte path of the script
script_folder = os.path.dirname(script_path)
cwd = os.getcwd() # current working directory path
relative_folder_path = os.relpath(cwd, script_folder) # relative path from cwd to the folder containing the script I run

print("Relative folder path from cwd to script's folder:", relative_folder_path)

