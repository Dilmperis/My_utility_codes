# Example of a string:
filename = "17-03-30_12-53-58_122500000_182500000_bbox.npy"

#Get the last part:
has_bbox = filename.endswith('.npy') # boolean (True)
# or
ext = filename.split('.')[-1]  # 'npy' // Get the actual extension

# Check if "_bbox" is in the filename
has_bbox = "_bbox" in filename
