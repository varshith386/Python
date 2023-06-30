import torch.nn as nn
from torchsummary import summary

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,4,kernel_size=5,stride=2),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(4,8,kernel_size=5,stride=2),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(8,16,kernel_size=3,stride=2),nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(16,64,kernel_size=3,stride=2),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=2),nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,stride=2),nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=2),nn.ReLU(inplace=True))
        self.pack = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7)

        self.linear = nn.Sequential(nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,64),nn.ReLU(inplace=True),nn.Linear(64,4))
        

    def forward(self, img):
        img = self.pack(img)
        img = img.view(img.size(0), -1)
        img = self.linear(img)
        img = nn.functional.log_softmax(img,dim=1)
        return img
    
    
if __name__ == "__main__":
    net = Network()
    summary(net, (3, 300, 300),device="cpu")