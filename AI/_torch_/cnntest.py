import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 0
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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
print(images.shape)
x = conv1(images)
print(x.shape)