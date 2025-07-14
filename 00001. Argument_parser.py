import numpy as np
import argparse
import pathlib as Path

'''
This code uses argument parser to parse arguments through command line (paths, epochw, output directory,... etc)
10/8/2024
'''

def parse_arguments():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description='Load the dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The path to the dataset directory')
    
    return parser.parse_args()


##########################################################################################
# Or more effeciently:
'''14/7/2025'''

parser = argparse.ArgumentParser(description="Train a model")

parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.lr}")
print(f"Use GPU: {args.use_gpu}")
