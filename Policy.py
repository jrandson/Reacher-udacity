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
        self.conv_layer1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x16
        self.conv_layer2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.size1d = 4*14

        # two fully connected layer
        self.fc1 = nn.Linear(self.size1d, 256)
        self.fc2 = nn.Linear(256, 4)

        # Sigmoid to
        self.sig = nn.Sigmoid()

        self.conv_layer3 = nn.Conv1d(1, 4, kernel_size=6, stride=2, bias=False)

    def forward(self, x):

        x = F.relu(self.conv_layer3(x))

        # flatten the tensor
        x = x.view(-1, self.size1d)
        x = F.relu(self.fc1(x))
        output = self.sig(self.fc2(x))

        return output


