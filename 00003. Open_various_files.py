''' This file contains lines of code for OPENING various types of files || 10.9.2024 ||'''


# .zip ################################################################################################
''' If you don't have enough space to unzip a big folder (meaning to unzip, have the folder TWICE,
and cause memory storage problems, then keep it zipped and unzip it in python!'''
# import zipfile
# import numpy as np

# file_path = '/media/etro/Seagate/guest01-auto_recordings_v2.zip'
# with zipfile.ZipFile(file_path) as zf:
#     subfolders = sorted({name.split('/')[1] for name in zf.namelist() if name.count('/') > 1})
#     print(subfolders)


# .npy ################################################################################################
# import numpy as np
# file_path = #add npy file here
# data = np.load(file_path)
# print("Shape of the array:", data.shape)
# print('data.dtype:', data.dtype) 
# print(data[:10])

# .bson ################################################################################################
# import bson
# bson_file_path = #add bson file here
# with open(bson_file_path, "rb") as f:
#     data = bson.decode_all(f.read())
# print(type(data))

# .aedat4 ################################################################################################
# import dv
# aedat4_file_path = #add aedat4 file here
# with dv.AedatFile(aedat4_file_path) as f:
#     print(f.names) # That's a list with the relevant info that the aedat4 contains

# .h5 ################################################################################################
# import h5py
# import hdf5plugin
# file_path = #add h5py file here 
# with h5py.File(file_path, 'r') as f:
#     print("Available datasets: ", list(f.keys())) # List all datasets in the file! keys() works only for h5py._hl.group.Group
#     data = f['dataset_name']  # Replace 'dataset_name' with the name of the dataset you want to load
#     data = f.get('dataset_name') # THis works the same ways as data = f['dataset_name'] 
#     print("Shape of the dataset:", data.shape)

# EXTRA CODE FROM OTHERS:
# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array

# .npz ################################################################################################
# import numpy as np
# file_path = #add npz file here
# data = np.load(file_path, allow_pickle=True)
# print(f"Contains {len(data.files)} files.")
# for array_name in data.files:
#         array_data = data[array_name]
#         print(f"\nArray '{array_name}':")
#         print(f"Shape: {array_data.shape}")
#         print(f"Data type: {array_data.dtype}")
#         # print(f"Array content (sample): {array_data if array_data.size < 10 else array_data[:10]}")  # Show first 10 elements if large
        

# .tar ################################################################################################  "tar" stands for "tape archive", as it was initially developed for writing data to tape drives.
# tar files can be extracted in local folders!
# import tarfile

# file_path = #add tar file here
# with tarfile.open(file_path, mode='r', ) as file:
#     print("Files in the archive:")
#     i = 0
#     for member in file.getmembers():
#         # print(member.name)
#         i+=1
#     print(f'Number of folders: {i}')


#  .list file extension that is basically the same as JSON format but implemented using newline characters to separate JSON values
#  A file with the .list extension is an APT list file used in the Debian (Linux) operating system. 
#  It contains a collection of software package download sources. They're created by the included APT (Advanced Package Tool). 



# .jsonl ################################################################################################ 
# (It means json + lines. It is the same as JSON format but implemented using newline characters to separate JSON values)


