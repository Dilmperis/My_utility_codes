import os

path_folder = #add path here


def print_tree(root_folder, indent=""):
    # Print the root folder name first
    folder_name = os.path.basename(root_folder)
    print(folder_name)
    
    # Get list of files and directories
    items = os.listdir(root_folder)
    
    for index, item in enumerate(items):
        path = os.path.join(root_folder, item)
        # Check if it's the last item to adjust tree branches
        if index == len(items) - 1:
            print(indent + "└── " + item)
            new_indent = indent + "    "  # add empty spaces for last item
        else:
            print(indent + "├── " + item)
            new_indent = indent + "│   "  # add pipe for non-last items

        # If it's a folder, recurse into it
        if os.path.isdir(path):
            print_tree(path, new_indent)


print_tree(path_folder)
