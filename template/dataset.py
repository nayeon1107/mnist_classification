
# import some packages you need here
import os
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, data_type):

        # write your codes here
        self.data_dir = data_dir
        self.data_type = data_type

        data = os.path.join(data_dir, data_type)
        filenames = os.listdir(data)
        self.full_filenames = [os.path.join(data, f) for f in filenames]
        self.transform = transforms.Compose([
            transforms.ToTensor(), # [0,1]
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):

        # write your codes here
        return len(self.full_filenames)

    def __getitem__(self, idx):

        # write your codes here
        img_name = os.path.join(self.data_dir, self.data_type, self.full_filenames[idx])
        img = Image.open(img_name)
        img = self.transform(img)
        label = int(img_name.split('.')[0][-1])

        return img, label

if __name__ == '__main__':

    # write test codes to verify your implementations
    data_dir = "/home/user/Desktop/na/mnist-classification/data"
    data_type = "train"
    mnist_dataset = MNIST(data_dir, data_type)
    image, label = mnist_dataset[1]

    print("Image shape:", image.shape)
    print("Label:", label)