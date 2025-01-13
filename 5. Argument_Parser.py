import argparse

parser = argparse.ArgumentParser() # Initialize the parser with ArgumentParser

# Pass training parameters through command line with add_argument
parser.add_argument('-e', '--epochs', type=int, required= True, help='Number of epochs')
parser.add_argument('-lr', '--learning_rate', type=float, required=True, help='learning rate')
parser.add_argument('-t', type=float, required=True, help='random parameter')
# etc....

args = parser.parse_args() # Get the arguments 

# Access the parameters:
epochs = args.epochs   # Not args.e, use the long name 
lr = args.learning_rate  # Not args.lr, use the long name 
tt = args.t 

print(epochs)
print(lr)
print(tt)

'''
Example of command in use:

CUDA_VISIBLE_DEVICES=0 python3 my_script.py -e=10 -lr=0.01 -t=0.1 
'''
