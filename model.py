import os
import torch
from torch import nn
class_dim=10
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=5,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(5568,1024),nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,class_dim) 
        )
    def forward(self,x):
        return self.net(x)
if __name__ =='__main__':
    X=torch.randn(1,1,128,22)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)
