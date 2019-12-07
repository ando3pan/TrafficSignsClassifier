import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import transforms, utils
import pickle
import os

class loader(Dataset):
    def __init__(self, pickle_file, transform=None):
        self.transform = transform
        dataset = pickle.load(open(pickle_file, 'rb'))
        self.images = dataset['features']
        self.labels = dataset['labels']
#         self.root_dir = root_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])       
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    
class Nnet(nn.Module):
    def __init__(self):
        super(Nnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64 , 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64 , 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU( inplace=True),
            
            nn.Conv2d(64, 64, 5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU( inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 9, 300),
#             nn.ReLU( inplace=True),
#             nn.Linear(4500, 2500),
#             nn.ReLU( inplace=True),
#             nn.Linear(2500, 1000),
            nn.ReLU( inplace=True),
            nn.Linear(300, 200),
            nn.ReLU( inplace=True),
            nn.Linear(200, 100),
            nn.ReLU( inplace=True),
            nn.Linear(100, 44),
            
            #COMMENT SOFTMAX DURING TRAINING
            #UNCOMMENT DURING TEST
#             nn.Softmax()
        )

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features

    def forward(self, input):
        x=self.main(input)
        x=x.view(-1, self.num_flat_features(x))
        return self.fc(x)

        
