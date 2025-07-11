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
