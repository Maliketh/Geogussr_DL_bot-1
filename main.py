import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from dataset import GeoGuessrDataset, train_transform, val_test_transform
from loss import VectorizedGeoLoss
from model import build_model
from utils import compute_distance_matrix, compute_geodesic_error
from train import train_model


full_dataset = pd.read_csv("train.csv")
num_samples = len(full_dataset)
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

train_dataset = GeoGuessrDataset("train.csv", "train_images", transform=train_transform)
val_dataset = GeoGuessrDataset("train.csv", "train_images", transform=val_test_transform)
test_dataset = GeoGuessrDataset("train.csv", "train_images", transform=val_test_transform)

indices = np.arange(num_samples)
np.random.shuffle(indices)
train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

train_dataset = Subset(train_dataset, train_idx)
val_dataset = Subset(val_dataset, val_idx)
test_dataset = Subset(test_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


centroids = pd.read_csv("country_centroids.csv") #You need to create a file with the centroids of each country on the dataset
country_list = sorted(centroids['country_code'].unique())
num_countries = len(country_list)

dist_matrix = compute_distance_matrix(centroids, country_list)
max_dist = dist_matrix.max()
sim_matrix = 1 - dist_matrix / max_dist
sim_matrix = torch.tensor(sim_matrix, dtype=torch.float32)

# Calculate weights of each country, give more weight to sparser countries
class_counts = full_dataset['country_code'].value_counts()
weights_array = np.zeros(num_countries, dtype=np.float32)
for i, c in enumerate(country_list):
    n_i = class_counts.get(c, 1)
    weights_array[i] = 1.0 / np.sqrt(n_i)
class_weights = torch.tensor(weights_array, dtype=torch.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(num_countries).to(device)
criterion = VectorizedGeoLoss(sim_matrix, class_weights)
optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

train_model(model, criterion, optimizer, scheduler,
            train_loader, val_loader, train_dataset, val_dataset,
            centroids, country_list, compute_geodesic_error,
            num_epochs=20, device=device)


print("\n Evaluating on test set...")
model.load_state_dict(torch.load("best_model.pth"))  # load best saved model
model.eval()
test_loss, all_preds, all_labels = 0.0, [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_dataset)
geo_error = compute_geodesic_error(all_preds, all_labels, centroids, country_list)
print(f"Test Loss: {test_loss:.4f}, Avg Geodesic Error: {geo_error:.2f} km")
