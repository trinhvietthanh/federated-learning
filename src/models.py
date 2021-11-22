#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in=784, dim_hidden=200, dim_out=10,
                 use_dropout=False, activation='ReLU'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_hidden),
                                     nn.Linear(dim_hidden, dim_hidden),
                                     nn.Linear(dim_hidden, dim_out)])
        self.softmax = nn.Softmax(dim=1)

        self.relu = getattr(nn, activation)()
        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout()

    def forward(self, x):                                       # input x of size (batchsize, 1, H, W)
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # x of size (batchsize, H*W)

        # Compute the forward propagation
        for layer in self.layers:
            x = layer(x)
            if self.dropout != None:
                x = self.dropout(x)
            x = self.relu(x)

        return self.softmax(x)

# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(512, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_hidden_channels1,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(args.num_hidden_channels1, args.num_hidden_channels2,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        
        self.fc1 = nn.Linear(args.num_hidden_channels2*7*7, 512)
        self.fc2 = nn.Linear(512, args.num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # x of size (batchsize, H*W)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


# ==========================================================================================================
# Main function
# ==========================================================================================================
if __name__ == "__main__":
    import torch
    from collections import namedtuple
    args = {'num_classes': 10, 'num_channels': 1,
            'num_hidden_channels1': 32, 'num_hidden_channels2': 64}
    args = namedtuple('x', args.keys())(*args.values())
    model = CNNMnist(args)

    x = torch.rand((2, 1, 28, 28))
    # model = MLP()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    # y = model(x)