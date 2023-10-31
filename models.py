## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv32 = nn.Conv2d(1, 32, 5) 
        self.conv64 = nn.Conv2d(32, 64, 5)
        self.conv128 = nn.Conv2d(64, 128, 5)
        
        self.pool = nn.MaxPool2d(2,2)
        
        #(64*108*108)
        self.linear1 = nn.Linear(73728, 1500)
        self.linear2 = nn.Linear(1500, 500)
        self.linear3 = nn.Linear(500, 136)
        self.dropout = nn.Dropout(0.3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv32(x)))
        x = self.pool(F.relu(self.conv64(x)))
        x = self.pool(F.relu(self.conv128(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        #print(x.size())  # Add this line to check the size of x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)  # Keypoint regression, so no activation function here
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
