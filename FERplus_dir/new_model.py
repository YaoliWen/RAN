import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    def __init__(self, model_dim, key_dim):
        super(Attention, self).__init__()
        self.linear_key = nn.Linear(model_dim, key_dim, bias=False)
        self.linear_query = nn.Linear(model_dim, key_dim, bias=False)
        self.softmax = nn.Softmax(dim = -1)
        self.model_dim = model_dim

    def forward(self, entire_img, org_img):
        value = []
        entire_key = self.linear_key(entire_img)
        org_query = self.linear_query(org_img)
        org_query = org_query.unsqueeze(1)
        # for i in range(6):
            # value.append(torch.matmul(entire_key[:,i,:], org_query))
        attention_weight = torch.matmul(entire_key, org_query.transpose(1, 2))
        attention_weight = attention_weight.squeeze(2)
        attention_weight = self.softmax(attention_weight)
        return attention_weight


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, end2end=True):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  #(224-6+6)/2=112 64*112*112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (112-2+2)/2=56 64*56*56
        self.layer1 = self._make_layer(block, 64, layers[0]) # 64*56*56->64*56*56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 128*28*28->128*28*28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 256*14*14—>256*14*14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 512*7*7->512*7*7
        self.avgpool = nn.AdaptiveAvgPool2d(1) # B*512*1*1
        self.alpha = Attention(512, 128)
        self.beta = Attention(1024, 128)
        self.fc = nn.Linear(1024,8)
        self.softmax = nn.Softmax(-1)

        #self.threedmm_layer = threeDMM_model(alfa,threed_model_data)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print 'input image shape',x.shape
        vs = []
        alphas = []
        for i in range(6):
            f = x[:,:,:,:,i]

            f = self.conv1(f) # B*W*H*F
            f = self.bn1(f)
            f = self.relu(f)
            f = self.maxpool(f)
        
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f) 
            print("layer4:  {}".format(f.shape()))
            f = self.avgpool(f) # B*512*1*1
            f = f.squeeze(3).squeeze(2) # B*512
            #MN_MODEL
            vs.append(f) #6*B*512
            alphas.append(self.alpha(f)) # 6*B*1
        vs_stack = torch.stack(vs, dim=1) # B*6*512
        global_face = vs_stack[:,0,:]
        alphas = self.alpha(vs_stack, global_face) # B*6
        alphas_part_max = alphas[:,1:6].max(dim=1) # B,B
        alphas_org = alphas[:,0] # B
        vm = vs_stack.mul(alphas.unsqueeze(2)).sum(1) # B*512
        for i in range(len(vs)): #6
            vs[i] = torch.cat([vs[i], vm], dim=1) # B*1024(6*B*1024)
        vs_stack_1024 = torch.stack(vs, dim=1) # B*6*1024
        org_face_1024 = vs_stack_1024[:,0,:]
        betas = self.beta(vs_stack_1024, org_face_1024) #B*6
        out = vs_stack_1024.mul((betas*alphas).unsqueeze(2)).sum(1).div((betas*alphas).sum(1).unsqueeze(1)) # B*1024
        pred_score = self.fc(out) # B*8
        # pred_score = self.softmax(pred_score)
        return pred_score, alphas_part_max, alphas_org


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50( **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    #if pretrained:
     #   model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
class MyLoss(nn.Module):    
    def __init__(self):        
        super(MyLoss, self).__init__()          
    def forward(self, alphas_part_max, alphas_org):
        size = alphas_org.shape[0]
        loss_wt = 0.0
        for i in range(size):
            loss_wt += max(torch.Tensor([0]).cuda(), 0.1 - (alphas_part_max[i] - alphas_org[i]))       
        return  loss_wt/size
