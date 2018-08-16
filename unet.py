# coding: utf-8

"""
pytorch implementation of U-net from https://github.com/jaxony/unet-pytorch
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

# image size after confolution
# W=(W−F+2P)/S+1 
# f is the receptive field (filter width), 
# p is the padding 
# s is the stride

def conv3x3(in_ch, out_ch, stride=1, padding=1, bias=True, groups=1):
    """
    default parameter does not change the image size
    """
    return nn.Conv2d(
       in_ch,
       out_ch,
       kernel_size=3,
       stride=stride,
       padding=padding,
       bias=bias,
       groups=groups 
    )


def upconv2x2(in_ch, out_ch, mode='transpose'):
    """mode is either 'transpose' or 'bilinear'"""
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )
    elif mode == 'bilinear':
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_ch, out_ch)
        )
    else:
        raise NotImplementedError("%s mode is not implemented" % mode) 

def conv1x1(in_ch, out_ch, groups=1):
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=1,
        groups=groups,
        stride=1
    )

class DownConv(nn.Module):
    """Helper module that performs 2 convolution and 1 maxpooling
    added Batch normalization layer in the original code.
    """

    def __init__(self, in_ch, out_ch, pooling=True):
        super(DownConv, self).__init__()
        self.pooling = pooling 

        self.conv = nn.Sequential(
            conv3x3(in_ch, out_ch, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            conv3x3(out_ch, out_ch, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        before_pool = self.conv(x)

        if self.pooling:
            x = self.pool(before_pool)
        else:
            x = before_pool
        
        return x, before_pool


class UpConv(nn.Module):
    """A helper module that performs 2 conv and 1 up-convolution.

    """
    def __init__(self, in_ch, out_ch, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(in_ch, out_ch, up_mode)

        if merge_mode == 'concat':
            conv1 = conv3x3(2 * out_ch, out_ch)
        elif merge_mode == 'add':
            conv1 = conv3x3(out_ch, out_ch)
        else:
            raise NotImplementedError("merge_mode %s is not implemented" % merge_mode)

        self.conv = nn.Sequential(
            conv1,
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ) 
    
    def forward(self, x_en, x_de):
        """
        x_en: from encoder path
        x_de: from decoder path
        """

        x_de = self.upconv(x_de)

        if self.merge_mode == 'concat':
            x = torch.cat((x_de, x_en), 1)
        else:
            x = x_en + x_de
        
        x = self.conv(x)
        
        return x

    
class Resizing:
    def __init__(self, left, right, top, bottom):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
        self.m = nn.ReflectionPad2d((left, right, top, bottom))
        
    def pad(self, imgs):
        return self.m(imgs)
    
    def unpad(self, imgs):
        return imgs[:, :, self.top:-self.bottom, self.left:-self.right]


class UNet(nn.Module):

    def __init__(self, num_class, input_channel, depth, start_filters=64,
                up_mode='transpose', merge_mode='concat',
                pad_l=0, pad_r=0, pad_t=0, pad_b=0):
        super(UNet, self).__init__()
        
        #self.pad_l = pad_l
        #self.pad_r = pad_r
        #self.pad_t = pad_t
        #self.pad_b = pad_b
        self.rs = Resizing(pad_l, pad_r, pad_t, pad_b)

        if up_mode == 'bilinear' and merge_mode == 'add':
            raise ValueError("up_mode \"bilinear\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
    


        down_convs = []
        up_convs = []

        # creating encoder
        for i in range(depth):
            out_ch = start_filters * 2**i

            if i == 0:
                in_ch = input_channel
            else:
                in_ch = start_filters * (2 ** (i-1))

            pooling = True if i < depth - 1 else False

            down_conv = DownConv(in_ch, out_ch, pooling=pooling)
            down_convs.append(down_conv)
    
        for i in range(depth - 1):
            in_ch = out_ch
            out_ch = in_ch // 2

            up_conv = UpConv(in_ch, out_ch, up_mode=up_mode, merge_mode=merge_mode)
            up_convs.append(up_conv)
        
        self.conv_final = conv1x1(out_ch, num_class)

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
    
    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)
    
    def forward(self, x, print_size=False):
        x = self.rs.pad(x)
        
        if print_size:
            print("first", x.size())

        encoder_outs = []

        for down_conv in self.down_convs:
            x, before_pool = down_conv(x)
            encoder_outs.append(before_pool)

            if print_size:
                print("down", x.size())
        
        for i, up_conv in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = up_conv(before_pool, x)

            if print_size:
                print("up", x.size())

        
        x = self.conv_final(x)
        
        if print_size:
                print("final", x.size())

        x = F.sigmoid(x)
        
        x = self.rs.unpad(x)
        
        return x

if __name__ == "__main__":
    """
    testin
    """
    in_ch = 1
    n_class = 1
    model = UNet(num_class=n_class, input_channel=in_ch,  depth=5, merge_mode='concat', start_filters=32,
                pad_l=13, pad_r=14, pad_t=13, pad_b=14)
    x = Variable(torch.FloatTensor(np.random.random((1, 1, 101, 101))))
    model.forward(x, print_size=True)







            
            


