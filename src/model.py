import torch.nn as nn
    
class LDQN(nn.Module):
    def __init__(self, input, output):
        super(LDQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*input, 128)
        self.fc2 = nn.Linear(128, output)

        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
class MDQN(nn.Module):
    def __init__(self, input, output):
        super(MDQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128*input, 256)
        self.fc2 = nn.Linear(256, output)

        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class HDQN(nn.Module):
    def __init__(self, input, output):
        super(HDQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(256*input, 512)
        self.fc2 = nn.Linear(512, output)

        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x