import csv
import os
import random

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

train_folder = r'C:\Users\Katyal Hina\PycharmProjects\individualProject\train'
test_folder = r'C:\Users\Katyal Hina\PycharmProjects\individualProject\test'


def build_csv(directory_string, output_csv_name):
    directory = directory_string
    class_lst = os.listdir(directory)
    class_lst.sort()
    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index'])
        for name in class_lst:
            class_path = os.path.join(directory, name)
            file_list = os.listdir(class_path)
            for file_name in file_list:
                file_path = os.path.join(directory, name, file_name)
                writer.writerow([file_name, file_path, name, class_lst.index(name)])
    return


build_csv(train_folder, 'train.csv')
build_csv(test_folder, 'test.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

class_names = list(train_df['class_name'].unique())


class IntelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir="", transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        class_index = self.data.iloc[idx, 3]
        image = np.array(PIL.Image.open(image_path))  # Convert to numpy array
        if self.transform:
            image = PIL.Image.fromarray(image)  # Convert numpy array to PIL image
            image = self.transform(image)
        return image, class_index


train_dataset_untransformed = IntelDataset(csv_file='train.csv', root_dir="", transform=None)

# Visualize 10 random images from the loaded dataset
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = random.randint(0, len(train_dataset_untransformed))
    image, class_index = train_dataset_untransformed[idx]
    ax = plt.subplot(2, 5, i + 1)
    ax.title.set_text(str(class_names[class_index]))
    plt.imshow(image)

# Create a transform pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create datasets with transforms
train_dataset = IntelDataset(csv_file='train.csv', root_dir="", transform=image_transform)
test_dataset = IntelDataset(csv_file='test.csv', root_dir="", transform=image_transform)

# Visualize 10 random images from the loaded transformed train_dataset
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = random.randint(0, len(train_dataset))
    image, class_index = train_dataset[idx]
    class_name = class_names[class_index]
    ax = plt.subplot(2, 5, i + 1)
    ax.title.set_text(f"{class_name}-{class_index}")
    plt.imshow(image.permute(1, 2, 0))


class ImageClassificationBase(nn.Module):

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy_custom(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


import torch
import torch.nn as nn


class NaturalSceneClassification(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def __init__(self):
        super(NaturalSceneClassification, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Forward a dummy input to get the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.network(dummy_input)
        last_output_size = dummy_output.numel() / dummy_output.shape[0]

        self.fc_layers = nn.Sequential(
            nn.Linear(int(last_output_size), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, len(class_names))
        )

    def forward(self, xb):
        out = self.network(xb)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out


my_model = NaturalSceneClassification()


def accuracy_custom(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    val_losses = []
    val_accs = []
    for batch in val_loader:
        outputs = model(batch[0])
        loss = F.cross_entropy(outputs, batch[1])
        acc = accuracy_custom(outputs, batch[1])
        val_losses.append(loss.detach())
        val_accs.append(acc)
    return {'val_loss': torch.stack(val_losses).mean().item(), 'val_acc': torch.stack(val_accs).mean().item()}


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


num_epochs = 30
opt_func = torch.optim.Adam
lr = 0.001

# Fitting the model on training data and recording the result after each epoch
history = fit(num_epochs, lr, my_model, DataLoader(train_dataset, batch_size=32),
              DataLoader(test_dataset, batch_size=32), opt_func)


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


fig, ax = plt.subplots(2, 1, figsize=(10, 8))

plot_accuracies(history)
plot_losses(history)

fig.savefig("metrics.png")  # Save figure locally as PNG
plt.show()  # Show figure

image, class_index = train_dataset[0]
predictions = my_model(image.unsqueeze(0))
score = torch.nn.functional.softmax(predictions, dim=1)
predicted_class = torch.argmax(score)
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[predicted_class], 100 * score[0, predicted_class]))