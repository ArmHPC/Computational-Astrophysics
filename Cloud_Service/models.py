from torch import flatten

import torch.nn as nn
import torch.nn.functional as F

from MobileNetV2 import MobileNetV2
from ResNet import ResNet


class FeatureBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1_conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        out += x
        return F.relu(out)


class ClassifierDefault(nn.Module):
    def __init__(self, class_count, input_shape, n_channels=1, depth_1=128, depth_2=64, kernel_size=3):
        super(ClassifierDefault, self).__init__()

        self.ks = kernel_size

        self.conv1 = nn.Conv2d(n_channels, depth_1, kernel_size=self.ks, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(depth_1, depth_1, kernel_size=self.ks, padding=1)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv3 = nn.Conv2d(depth_1, depth_2, kernel_size=self.ks, padding=1)
        self.relu3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv4 = nn.Conv2d(depth_2, depth_2, kernel_size=self.ks, padding=1)
        self.relu4 = nn.ReLU()
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(depth_2, depth_2, kernel_size=self.ks, padding=1)
        self.relu5 = nn.ReLU()
        self.mp5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3840, 512)
        self.relu6 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, class_count)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mp3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.mp4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.mp5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape=(160, 50)):
        super().__init__()

        height, width = input_shape
        num_max_pools = 4
        max_pool_ks = 2

        depth_0 = 1
        depth_1 = 32
        depth_2 = 64
        depth_3 = 128
        depth_4 = 256
        depth_fm = 256

        drop_prob_1 = 0.2
        drop_prob_2 = 0.3
        drop_prob_3 = 0.5

        neurons_1 = 256
        neurons_2 = num_classes

        self.act1 = nn.LeakyReLU(negative_slope=0.05)
        self.act2 = nn.Softmax(dim=1)

        self.conv11 = nn.Conv2d(in_channels=depth_0, out_channels=depth_1, kernel_size=3, padding='same')
        self.conv12 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, padding='same')
        self.mp1 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.bn1 = nn.BatchNorm2d(depth_1)
        self.drop1 = nn.Dropout(p=drop_prob_1)

        self.conv21 = nn.Conv2d(in_channels=depth_1, out_channels=depth_2, kernel_size=3, padding='same')
        self.conv22 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, padding='same')
        self.conv23 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, padding='same')
        self.mp2 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.bn2 = nn.BatchNorm2d(depth_2)
        self.drop2 = nn.Dropout(p=drop_prob_1)

        self.conv31 = nn.Conv2d(in_channels=depth_2, out_channels=depth_3, kernel_size=3, padding='same')
        self.conv32 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, padding='same')
        self.conv33 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, padding='same')
        self.mp3 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.bn3 = nn.BatchNorm2d(depth_3)

        # 4th block
        self.drop3 = nn.Dropout(p=drop_prob_1)

        self.conv41 = nn.Conv2d(in_channels=depth_3, out_channels=depth_4, kernel_size=3, padding='same')
        self.conv42 = nn.Conv2d(in_channels=depth_4, out_channels=depth_4, kernel_size=3, padding='same')
        self.conv43 = nn.Conv2d(in_channels=depth_4, out_channels=depth_4, kernel_size=3, padding='same')
        self.mp4 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.bn4 = nn.BatchNorm2d(depth_fm)

        fc_in_features = depth_fm * (height // (max_pool_ks ** num_max_pools)) * (
                width // (max_pool_ks ** num_max_pools))
        self.drop3 = nn.Dropout(p=drop_prob_2)
        self.fc1 = nn.Linear(in_features=fc_in_features, out_features=neurons_1)
        self.drop4 = nn.Dropout(p=drop_prob_3)
        self.fc2 = nn.Linear(in_features=neurons_1, out_features=neurons_2)

    def forward(self, x):
        # First block
        x = self.act1(self.conv11(x))
        x = self.act1(self.conv12(x))
        x = self.act1(self.conv13(x))
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        # Second block
        x = self.act1(self.conv21(x))
        x = self.act1(self.conv22(x))
        x = self.act1(self.conv23(x))
        x = self.mp2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        # Third block
        x = self.act1(self.conv31(x))
        x = self.act1(self.conv32(x))
        x = self.act1(self.conv33(x))
        x = self.mp3(x)
        x = self.bn3(x)

        # Fourth block
        x = self.act1(self.conv41(x))
        x = self.act1(self.conv42(x))
        x = self.act1(self.conv43(x))
        x = self.mp4(x)
        x = self.bn4(x)

        # Fully-Connected
        x = flatten(x, 1)
        x = self.drop3(x)
        x = self.act1(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        # x = self.act2(x)

        return x


class ClassifierBN(nn.Module):
    def __init__(self, num_classes, input_shape=(160, 50)):
        super().__init__()

        height, width = input_shape
        num_max_pools = 4
        max_pool_ks = 2

        depth_0 = 1
        depth_1 = 32
        depth_2 = 64
        depth_3 = 128
        depth_4 = 256
        depth_fm = 256

        drop_prob_1 = 0.2
        drop_prob_2 = 0.3
        drop_prob_3 = 0.5

        neurons_1 = 256
        neurons_2 = num_classes

        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=False)

        self.conv11 = nn.Conv2d(in_channels=depth_0, out_channels=depth_1, kernel_size=3, padding='same')
        self.bn11 = nn.BatchNorm2d(depth_1)
        self.conv12 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, padding='same')
        self.bn12 = nn.BatchNorm2d(depth_1)
        self.conv13 = nn.Conv2d(in_channels=depth_1, out_channels=depth_1, kernel_size=3, padding='same')
        self.bn13 = nn.BatchNorm2d(depth_1)
        self.mp1 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.drop1 = nn.Dropout(p=drop_prob_1)

        self.conv21 = nn.Conv2d(in_channels=depth_1, out_channels=depth_2, kernel_size=3, padding='same')
        self.bn21 = nn.BatchNorm2d(depth_2)
        self.conv22 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, padding='same')
        self.bn22 = nn.BatchNorm2d(depth_2)
        self.conv23 = nn.Conv2d(in_channels=depth_2, out_channels=depth_2, kernel_size=3, padding='same')
        self.bn23 = nn.BatchNorm2d(depth_2)
        self.mp2 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.drop2 = nn.Dropout(p=drop_prob_1)

        self.conv31 = nn.Conv2d(in_channels=depth_2, out_channels=depth_3, kernel_size=3, padding='same')
        self.bn31 = nn.BatchNorm2d(depth_3)
        self.conv32 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, padding='same')
        self.bn32 = nn.BatchNorm2d(depth_3)
        self.conv33 = nn.Conv2d(in_channels=depth_3, out_channels=depth_3, kernel_size=3, padding='same')
        self.bn33 = nn.BatchNorm2d(depth_3)
        self.mp3 = nn.MaxPool2d(kernel_size=max_pool_ks)
        self.drop3 = nn.Dropout(p=drop_prob_1)

        # 4th block
        self.conv41 = nn.Conv2d(in_channels=depth_3, out_channels=depth_4, kernel_size=3, padding='same')
        self.bn41 = nn.BatchNorm2d(depth_4)
        self.conv42 = nn.Conv2d(in_channels=depth_4, out_channels=depth_4, kernel_size=3, padding='same')
        self.bn42 = nn.BatchNorm2d(depth_4)
        self.conv43 = nn.Conv2d(in_channels=depth_4, out_channels=depth_4, kernel_size=3, padding='same')
        self.bn43 = nn.BatchNorm2d(depth_4)
        self.mp4 = nn.MaxPool2d(kernel_size=max_pool_ks)

        fc_in_features = depth_fm * (height // (max_pool_ks ** num_max_pools)) * (
                width // (max_pool_ks ** num_max_pools))
        self.drop3 = nn.Dropout(p=drop_prob_2)
        self.fc1 = nn.Linear(in_features=fc_in_features, out_features=neurons_1)
        self.drop4 = nn.Dropout(p=drop_prob_3)
        self.fc2 = nn.Linear(in_features=neurons_1, out_features=neurons_2)

    def forward(self, x):
        # First block
        x = self.act(self.bn11(self.conv11(x)))
        x = self.act(self.bn12(self.conv12(x)))
        x = self.act(self.bn13(self.conv13(x)))
        x = self.mp1(x)
        x = self.drop1(x)

        # Second block
        x = self.act(self.bn21(self.conv21(x)))
        x = self.act(self.bn22(self.conv22(x)))
        x = self.act(self.bn23(self.conv23(x)))
        x = self.mp2(x)
        x = self.drop2(x)

        # Third block
        x = self.act(self.bn31(self.conv31(x)))
        x = self.act(self.bn32(self.conv32(x)))
        x = self.act(self.bn33(self.conv33(x)))
        x = self.mp3(x)

        # Fourth block
        x = self.act(self.bn41(self.conv41(x)))
        x = self.act(self.bn42(self.conv42(x)))
        x = self.act(self.bn43(self.conv43(x)))
        x = self.mp4(x)

        # Fully-Connected
        x = flatten(x, 1)
        x = self.drop3(x)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


class Model(nn.Module):
    def __init__(self, num_classes, input_shape, arch='default'):
        super(Model, self).__init__()

        assert arch in ['default', 'default_prev', 'default_bn', 'mobilenet', 'resnet']

        if arch == 'default':
            self.classifier = Classifier(num_classes, input_shape)
        elif arch == 'default_bn':
            self.classifier = ClassifierBN(num_classes=num_classes, input_shape=input_shape)
        elif arch == 'default_prev':
            self.classifier = ClassifierDefault(class_count=num_classes, input_shape=input_shape)
        elif arch == 'mobilenet':
            self.classifier = MobileNetV2(input_channel=1, n_classes=num_classes)
        else:
            self.classifier = ResNet(input_channel=1, n_classes=num_classes)

    def forward(self, input_i):
        res = self.classifier(input_i)
        return res
