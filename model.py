import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD, lr_scheduler
class EmbeddingNet(nn.Module):

  # extract embeddings from the input image 
  # passes through a convolutional layer and a fully-connected layer 

  # TODO: fix the inputs dimensions 

# nn.Conv3d(1, 32, 5)
# torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, 
# padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

# Parametric ReLU 
# https://medium.com/@shauryagoel/prelu-activation-e294bb21fefa
# LeakyReLU was introduced, which doesn’t zero out the negative inputs as ReLU does. 
# Instead, it multiplies the negative input by a small value (like 0.02) and keeps the positive input as is.
# We can learn the slope parameter using backpropagation at a negligible increase in the cost of training.

# torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)


#  input size (N, C_{in}, D, H, W)
# N = number of examples 
# 
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv3d(1, 5, 5), nn.PReLU())
                                    #  nn.MaxPool3d(2, stride=2))
                                    #   nn.Conv3d(32, 64, 5), nn.PReLU(),
                                    #   nn.MaxPool3d(2, stride=2)) 

        self.fc = nn.Sequential(nn.Linear(64 * 22 * 51 * 51, 256), 
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 64) 
                                )

    #@torch.autocast(device_type='cpu', dtype=torch.double)
    def forward(self, x):
        
        x = torch.unsqueeze(x, 1)
        output = self.convnet(Variable(x.type(torch.double))) # something is wrong here
        output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)