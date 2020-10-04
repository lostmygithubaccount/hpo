# imports
import os
import torch
import mlflow
import argparse
import torchvision

import torch.optim as optim
import torchvision.transforms as transforms

from model import Net

# arg parser
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, help="Path to the training data")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for SGD")
parser.add_argument("--mu", type=float, default=0.9, help="Momentum for SGD")

args = parser.parse_args()

print("===== DATA =====")
print("DATA PATH: " + args.data_dir)
print("LIST FILES IN DATA PATH...")
print(os.listdir(args.data_dir))
print("================")

# prepare DataLoader for CIFAR10 data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
    root=args.data_dir, train=True, download=False, transform=transform,
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

# define convolutional network
net = Net()

# set up pytorch loss /  optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mu,)

# train the network
for epoch in range(args.epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # unpack the data
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            loss = running_loss / 2000
            mlflow.log_metric("loss", loss)  # log loss metric to AML
            print(f"epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}")
            running_loss = 0.0

print("Finished Training")
