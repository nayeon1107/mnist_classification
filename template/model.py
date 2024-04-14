import torch
import torch.nn as nn
from torchsummary import summary

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        # write your codes here
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fclayer1 = nn.Sequential(
            nn.Linear(256,120),
            # torch.nn.Dropout(0.5)
            nn.Linear(120,84),
            nn.ReLU(),
            # torch.nn.Dropout(0.5)
        )
        self.fclayer2 = nn.Sequential(
            nn.Linear(84,10)
        )
        
    def forward(self, img):
        # write your codes here
        img = self.layer1(img)
        img = self.layer2(img)
        img = torch.flatten(img,1)
        img = self.fclayer1(img)
        output = self.fclayer2(img)
        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):

        # write your codes here
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 56)
        self.fc2 = nn.Linear(56, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, img):

        # write your codes here
        img = torch.flatten(img,1)
        img = self.fc1(img)
        img = self.fc2(img)
        output = self.fc3(img)

        return output


if __name__ == '__main__' :
    model1 = LeNet5()
    model2 = CustomMLP()
    summary(model1, (1,28,28))
    summary(model2, (1,28,28))
    