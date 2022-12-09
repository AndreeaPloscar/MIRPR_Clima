import torch.nn as nn


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=(kernel_size, kernel_size),
                              out_channels=out_channels, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleNet, self).__init__()

        self.unit1 = Unit(in_channels=1, out_channels=32, kernel_size=1, padding=0)
        self.unit2 = Unit(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        # self.dropout1 = nn.Dropout2d()

        self.unit3 = Unit(in_channels=32, out_channels=64, kernel_size=3)
        # self.unit4 = Unit(in_channels=64, out_channels=64, kernel_size=3)
        # self.unit5 = Unit(in_channels=64, out_channels=64, kernel_size=3)

        self.dropout2 = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1, padding=1, dilation=1)

        self.unit6 = Unit(in_channels=64, out_channels=128, kernel_size=3)
        # self.unit7 = Unit(in_channels=128, out_channels=128, kernel_size=3)

        self.dropout3 = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1, padding=1, dilation=1)

        self.unit9 = Unit(in_channels=128, out_channels=256, kernel_size=3)
        self.unit10 = Unit(in_channels=256, out_channels=256, kernel_size=3)

        # self.dropout4 = nn.Dropout2d()
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=1, padding=1, dilation=1)

        self.unit15 = Unit(in_channels=256, out_channels=128, kernel_size=3)

        self.unit17 = Unit(in_channels=128, out_channels=64, kernel_size=3)


        self.unit19 = Unit(in_channels=64, out_channels=32, kernel_size=3)


        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2,  self.unit3, self.dropout2,
                                  self.pool1, self.unit6,
                                  self.pool2, self.unit9, self.unit10,
                                 self.pool3,
                                 # self.upsample1,
                                 # self.unit15,
                                 self.unit15, self.unit17,
                                 # self.upsample2,

                                 self.unit19,

                                 # self.unit21, self.unit22,
                                 # self.upsample3,
                                 # self.unit23,
                                 # self.unit24,
                                 # self.unit25, self.unit26
                                 )

        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 512)
        output = self.fc(output)
        return output
