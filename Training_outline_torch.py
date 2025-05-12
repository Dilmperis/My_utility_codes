from torch.utils.data import DataLoader, Dataset
import torch
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################################################
# 1. Dataset Class
class CustomDataset(Dataset):
    '''This class is responsible for loading and preprocessing data.'''
    def __init__(self, data_paths, labels, transforms=None): 
        #Initializes paths to data, any preprocessing steps, and other necessary variables.
        self.data_paths = data_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self): #Returns the size of the dataset.
        return len(self.data_paths)

    def __getitem__(self, idx):   
        '''Loads and returns a data sample and its label (or target). Handles transformations and data augmentation.'''
        data = load_data(self.data_paths[idx])
        label = self.labels[idx]

        if self.transforms:
            data = self.transforms(data)

        return data, label 

########################################################################################################
# 2. DataLoader function
def create_dataloader(dataset, batch_size=12, shuffle=True, collate_fn=None):
    '''
    Use collate_fn when:

    1) You have variable-sized data like images of different dimensions.
    2) You’re working with dictionaries (e.g., for object detection with bounding boxes and labels).
    3) You need to apply custom batching logic, such as padding sequences or resizing images.

    Notes:
    1) collate_fn should be a callable function.
    2) If not specified, DataLoader uses the default collate_fn, which attempts to stack items directly.
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Create dataset instances
train_dataset = CustomDataset(train_data_paths, labels=train_labels, transforms=train_transforms)
val_dataset = CustomDataset(val_data_paths, labels=val_labels, transforms=val_transforms)
test_dataset = CustomDataset(test_data_paths, transforms=test_transforms)

# 4. Create DataLoader instances for training, validation, and testing
batch_size = 12
collate_fn = lambda x: tuple(zip(*x))  # Custom collate_fn (if needed)

train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Example of iterating through a DataLoader
for batch_idx, (data, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {labels}")

########################################################################################################
# 3. Model + Training

# Define the Model Architecture
class CustomModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(CustomModel, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

########################################################################################################
# 4. Training
from torchvision.models import resnet18


# Initialize the model, loss function, optimizer, and scheduler
model = CustomModel(num_classes=10, pretrained=True)
model.to(device)

num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Initialize lists to save training and validation losses
train_losses = []
val_losses = []

# Training and Validation Loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()  # Sum loss over batch

    # Average train loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()   # Sum loss over batch

    # Average validation loss for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    # Print losses for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# Save the model on a chpt 
# save_path = .....
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': num_epochs,
#     'loss': epoch_train_loss / len(train_loader),
# }, save_path)


#  Save the Training and Validation Losses to a File
losses = {
    "train_losses": train_losses,
    "val_losses": val_losses}

with open("training_val_losses.json", "w") as f:
    json.dump(losses, f)

print("Training complete. Losses saved to 'training_val_losses.json'.")

########################################################################################################
# 5. Testing

# # Load the final saved model checkpoint
# checkpoint = torch.load('./model_final.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# print("Model loaded for testing.")


def test(model, test_dataset, batch_size=32, checkpoint_path=None, device=None, results_path="test_results.json"):
    """
    Tests a given model on a test dataset and saves results to a JSON file.
    
    Parameters:
    - model: PyTorch model to be tested.
    - test_dataset: Dataset object for the test set.
    - batch_size: Batch size for the DataLoader.
    - checkpoint_path: Path to the model checkpoint file. If provided, loads the model weights.
    - device: Device for computation (e.g., 'cuda' or 'cpu'). Defaults to CUDA if available.
    - results_path: Path to save test results JSON file.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint if provided
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Set up DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Testing loop
    all_predictions = []
    all_targets = []
    total_time = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Measure inference time !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            start_time = time.time()
            outputs = model(inputs)
            total_time += time.time() - start_time

            # Get predictions and calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            # Collect predictions and targets
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    
    # Calculate metrics
    accuracy = correct_predictions / total_samples
    avg_inference_time = total_time / len(test_loader)

    # Save results
    results = {
        "accuracy": accuracy,
        "average_inference_time": avg_inference_time,
        "predictions": all_predictions,
        "targets": all_targets
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f)

    print(f"Testing completed. Accuracy: {accuracy:.4f}, Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"Results saved to {results_path}")

# Usage example 
from your_model import YourModel  # Import your model class
from your_dataset import YourTestDataset  # Import your test dataset class

# Initialize model and dataset
model = YourModel()
test_dataset = YourTestDataset("path/to/test_data")

# Run the test
test(model, test_dataset, batch_size=32, checkpoint_path="path/to/checkpoint.pth", results_path="test_results.json")



