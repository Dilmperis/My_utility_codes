import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Which run to visualize?')
parser.add_argument('--run', '-r', type=str, help='Which number of run (1,2,3,...) to visualize?')
parser.add_argument('--calculate_mAP_every_5_epochs', '-c', 
                    type=bool, default=False,
                    help=("Is the dataset big? If so, the mAP is calculated every 5 epochs. \
                          The rest is always 0.0. I want to exclude the zeros from the plots."))
args = parser.parse_args()


print(f"Run: {args.run}, Calculate mAP every 5 epochs: {args.calculate_mAP_every_5_epochs}")
list_of_args = sys.argv
print(f"List of arguments: {list_of_args}")

# Save the command:
log_path_of_txt_file = os.path.join(os.getcwd(), 'abstarct_logging.txt')
with open(log_path_of_txt_file, 'a') as f: # Options: 'a', 'w', 'x', 'r' and their "+".... 
                                           # ALL CREATE THE FILE IF NOT EXIST, expect of 'r'
                                           # 'a' to append, 'w' to write, 'x' to create a new file, 'w+' to read and write
                                           # 'x+' to create a new file and read/write, 'a+' to append and read, '
                                           # 'r+' to read and write, 'r' to read
    f.write(f"command used: {sys.argv}\n") 
    f.write(f'This will add an extra line\n') # YOU CAN ADD MULTIPLE LINES!!!!!!

'''This will give you:
     command used: ['/media/etro/Seagate/guest01-auto_recordings_LATEST/absrtact.py', '-r=6', '-c=True']

   I don't likeit as a list... A better way is the next.
'''    

#=============================================================================================================

with open(log_path_of_txt_file, 'a') as f:
    f.write(f'command used: \n      python3 {" ".join(sys.argv)}\n') # This will give you:
    # command used: python3 /media/etro/Seagate/guest01-auto_recordings_LATEST/absrtact.py -r=6 -c=True
    
    # The " ".join(sys.argv) will take the elements of the list produced by sys.argv and
    # will add a space between them, creating a single string that represents the command line input.
