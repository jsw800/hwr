import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, channels, outputSize, inputSize=32, leakyRelu=False):
        super(CNN, self).__init__()

        ks = [3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1]
        nm = [64, 128, 128, 256, 256]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1, True)
        convRelu(2)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(3, True)
        convRelu(4)
        
        # self.fcin = nm[4] * ks[4] * ks[4]
        endsize = inputSize / 4
        self.fcin = endsize * endsize * 256
        self.fc1 = nn.Linear(self.fcin, nm[4] * ks[4] * ks[4] / 2)
        self.fc2 = nn.Linear(nm[4] * ks[4] * ks[4] / 2, outputSize)

        # cnn.add_module('fc1'.format(1),
        #     nn.Linear(nm[4] * ks[4] * ks[4], nm[4] * ks[4] * ks[4] / 2))

        # cnn.add_module('fc2'.format(1),
        #     nn.Linear(nm[4] * ks[4] * ks[4] / 2, outputSize))

        self.cnn = cnn
        self.softmax = nn.LogSoftmax()


    def forward(self, input):
        output = self.cnn(input)
        output = output.view(-1, self.fcin)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output


def create_model(config):
    return CNN(config['network']['num_of_channels'], len(config['classes']))
