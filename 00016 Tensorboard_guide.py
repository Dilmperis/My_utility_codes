import torch
from torch.utils.tensorboard import SummaryWriter
import random
import time
import sys
import argparse 

parser = argparse.ArgumentParser(description='TensorBoard Example')
parser.add_argument('--random_arg_for example', type=str, help='A random argument for demonstration purposes')

# Create a SummaryWriter to write logs for TensorBoard
writer = SummaryWriter(log_dir="runs_tensorboard/random_run")
writer.add_text('cmd', ' '.join(sys.argv)) # Sving the command line arguments to TensorBoard (at the text option of tensorboard)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = random.uniform(0.5, 2.0) - epoch * 0.2  # decreasing
    mAP = random.uniform(0.2, 0.8) + epoch * 0.05  # increasing
    
    # Log to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("mAP/val", mAP, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, mAP: {mAP:.4f}")
    time.sleep(0.5)  # just to simulate work

# Close the writer
writer.close()

'''
You start the tensorboard by using the command:

    tensorboard --logdir=file_path 
'''
