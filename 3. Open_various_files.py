''' This file contains lines of code for OPENING various types of files || 10.9.2024 ||'''

# .npy ################################################################################################
# import numpy as np
# file_path = #add npy file here
# data = np.load(file_path)
# print("Shape of the array:", data.shape)
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
#     print("Available datasets: ", list(f.keys())) # List all datasets in the file
#     data = f['dataset_name']  # Replace 'dataset_name' with the name of the dataset you want to load
#     print("Shape of the dataset:", data.shape)

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
# import tarfile

# file_path = #add tar file here
# with tarfile.open(file_path, mode='r', ) as file:
#     print("Files in the archive:")
#     i = 0
#     for member in file.getmembers():
#         # print(member.name)
#         i+=1
#     print(f'Number of folders: {i}')
