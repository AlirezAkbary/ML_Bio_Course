import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsampling=None, padding=0, use_batchnorm=True, use_dropout=True):
        super(ResidualBlock, self).__init__()

        self.downsampling = downsampling
        self.use_batch_norm = use_batchnorm
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size-1)//2)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if self.use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, (kernel_size-1)//2)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        result = self.conv1(x)
        if self.use_batch_norm:
            result = self.bn1(result)
        result = self.relu(result)
        if self.use_dropout:
            result = self.dropout(result)
        result = self.conv2(result)
        if self.use_batch_norm:
            result = self.bn2(result)

        if self.downsampling is not None:
            identity = self.downsampling(x)

        result += identity
        result = self.relu(result)
        if self.use_dropout:
            result = self.dropout(result)
        return result


class ResNet(nn.Module):
    def __init__(self, image_channels=3, num_classes=2, use_batchnorm=True, use_dropout=True):
        super(ResNet, self).__init__()
        self.current_input_channels = image_channels
        self.conv1 = nn.Conv2d(self.current_input_channels, 16, 7, 2, 3)
        self.current_input_channels = 16
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layer1 = self._make_residual_layer(16, 7, 2, use_batchnorm, use_dropout)
        self.res_layer2 = self._make_residual_layer(32, 5, 2, use_batchnorm, use_dropout)
        self.res_layer3 = self._make_residual_layer(64, 3, 2, use_batchnorm, use_dropout)
        #self.res_layer4 = self._make_residual_layer(64, 3, 2, use_batchnorm, use_dropout)
        self.pool_flatten = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(64, num_classes)

    def _make_residual_layer(self, out_channels, kernel_size, stride, use_batchnorm, use_dropout):
        downsampling = None
        if self.current_input_channels !=  out_channels or stride != 1:
            if use_batchnorm:
                downsampling = nn.Sequential(
                    nn.Conv2d(self.current_input_channels, out_channels, 1, stride, 0),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsampling = nn.Sequential(
                    nn.Conv2d(self.current_input_channels, out_channels, 1, stride, 0)
                )
        res_layer = ResidualBlock(self.current_input_channels, out_channels, kernel_size, stride, downsampling,
                                  use_batchnorm=use_batchnorm, use_dropout=use_dropout)
        self.current_input_channels = out_channels
        return res_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        #x = self.res_layer4(x)
        x = self.pool_flatten(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return x


