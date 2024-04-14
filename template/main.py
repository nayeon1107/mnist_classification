
import dataset
from model import LeNet5, CustomMLP

# import some packages you need here
import torch
import pickle
import argparse
import matplotlib.pyplot as plt


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
data_directory = "/home/user/Desktop/na/mnist-classification/data"

EPOCHS = 30

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("file_name")
args = parser.parse_args()
model_name = args.model
log_filename = args.file_name

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.to(device)
    model.train()
    trn_loss, acc = 0,0
    for image, label in trn_loader :
        image = image.to(device)
        label = label.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # loss & acc
        trn_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        acc += (predicted == label).sum().item()

    trn_loss /= len(trn_loader.dataset)
    acc /= len(trn_loader.dataset)

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    tst_loss, acc = 0,0
    with torch.no_grad() :
        for image, label in tst_loader :
            image = image.to(device)
            label = label.type(torch.LongTensor).to(device)
            output = model(image)
            loss = criterion(output, label)
            # loss & acc
            tst_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            acc += (predicted == label).sum().item()

    tst_loss /= len(tst_loader.dataset)
    acc /= len(tst_loader.dataset)

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    # write your codes here
    # 1) Dataset objects for training and test datasets
    dataset_train = dataset.MNIST(data_directory, "train")
    dataset_test = dataset.MNIST(data_directory, "test")

    # 2) DataLoaders for training and testing
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=256)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=256)

    # 3) model
    if model_name == 'LeNet5' :
        model = LeNet5().to(device)
    elif model_name == 'MLP' :
        model = CustomMLP().to(device)

    # 4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
    # 5) cost function: use torch.nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # train & test
    train_acc_all, train_loss_all, test_acc_all, test_loss_all = [], [], [], []
    
    print(f" === 📍 Applying model : {model_name} === ")
    for epoch in range(1,EPOCHS+1) :
        train_loss, train_acc = train(model, dataloader_train, device, criterion, optimizer)
        test_loss, test_acc = test(model, dataloader_test, device, criterion)
        train_acc_all.append(train_acc)
        train_loss_all.append(train_loss)
        test_acc_all.append(test_acc)
        test_loss_all.append(test_loss)

        print("Epoch:{}/{} | ".format(epoch, EPOCHS),
              "Train Accuracy: {:.4f}..".format(train_acc),
              "Train Loss: {:.4f}..".format(train_loss),
              "Test Accuracy: {:.4f}..".format(test_acc),
              "Test Loss: {:.4f}..".format(test_loss))

    return train_acc_all, train_loss_all, test_acc_all, test_loss_all

if __name__ == '__main__':
    train_acc, train_loss, test_acc, test_loss = main()
    pickle.dump({'train_acc' : train_acc,
                 'train_loss' : train_loss,
                 'test_acc' : test_acc,
                 'test_loss' : test_loss},
            open("/home/user/Desktop/na/mnist-classification"+f"/{log_filename}.pickle", "wb"))