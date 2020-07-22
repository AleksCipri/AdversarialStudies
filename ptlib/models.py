import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SimpleConvNet(nn.Module):
    '''sizes for 28x28 image'''

    def __init__(self, kaiming_normal=True):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        if kaiming_normal:
            init.kaiming_normal_(self.conv1.weight, a=0)
            init.kaiming_normal_(self.conv2.weight, a=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool(x))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class SimpleSGConvNet(nn.Module):
    '''sizes for 48x48 image'''

    def __init__(self, kaiming_normal=True):
        super(SimpleSGConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        if kaiming_normal:
            init.kaiming_normal_(self.conv1.weight, a=0)
            init.kaiming_normal_(self.conv2.weight, a=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool(x))
        x = x.view(-1, 32 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
