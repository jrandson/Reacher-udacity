import torch
import torch.nn as nn
import torch.nn.functional as F


# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        ########
        ##
        ## Modify your neural network
        ##
        ########

        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x16
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.size1d = (33 - 6) / 2 + 1

        # two fully connected layer
        self.fc1 = nn.Linear(self.size1d, 256)
        self.fc2 = nn.Linear(256, 4)

        # Sigmoid to
        self.sig = nn.Sigmoid()

        self.conv3 = nn.Conv1d(1, 4, kernel_size=6, stride=2, bias=False)

    def forward(self, x):
        ########
        ##
        ## Modify your neural network
        ##
        ########
        # x = F.relu(self.conv1(x))
        x = F.relu(self.conv3(x))
        # flatten the tensor
        x = x.view(-1, self.size1d)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))

