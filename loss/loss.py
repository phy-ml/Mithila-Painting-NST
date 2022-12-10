import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    def forward(self,x):
        # get the size of different dims
        b,c,h,w = x.size()

        # resize the input param
        resize_x = x.view(b,c,h*w)

        # perform batch multiplication of resized input and its transpose
        new_x = torch.bmm(resize_x, resize_x.transpose(1,2))

        new_x.div_(h*w)

        return new_x

class GramLoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix(x=input), target)
        return out

class ContentLoss(nn.Module):
    def forward(self,input, output):
        out = nn.MSELoss()(input, output)
        return out



