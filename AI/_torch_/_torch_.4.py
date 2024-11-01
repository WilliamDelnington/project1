import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 4
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="E:/Works/AI/images/data/fruits/fruits-360_dataset_100x100/Test",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="E:/Works/AI/images/data/fruits/fruits-360_dataset_100x100/Test",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

classes = (
    "apple", "avocado",
    "banana", "beetroot", "blueberry", 
    "cantaloupe", "carrot", "cherry", "cauliflower", "cocos", "corns", "cucumber",
    "eggplant", 
    "fig", 
    "grape", "ginger root", "grapefruit", "granadilla", 
    "kaki", "kiwi", "kohlrabi", "kumsquats"
    "lemon", "lime", "lychee",
    "mandarine", "mango", "mandarine", "mulberry", "melon", 
    "nectarine", "nut forest", 
    "orion", "orange", 
    "papaya", "passion fruit", "peach", "pear", "pineapple", "plum", "pomegranate", 
    "quince", 
    "rambutan", "raspberry", "redcurrant",
    "salak", "strawberry",
    "tangelo", "tomato",
    "walnut", "watermelon",
    "zucchini"
)

class ConvNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        pass
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
            
print("Finished Training")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    accuracy = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {accuracy} %")
    
    for i in range(len(classes)):
        accuracy = 100.0 * n_class_correct[i] * n_class_samples[i]
        print(f"Accuracy of class {classes[i]}: {accuracy} %")