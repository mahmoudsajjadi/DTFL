
import logging

import torch
import torch.nn as nn

import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, fedavg_base=False, tier=7, local_loss=False, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.local_loss = local_loss
        self.fedavg_base = fedavg_base
        self.local_v2 = False
        if kwargs:
            self.local_v2 = kwargs['local_v2']

        self.tier = tier  

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # initialization is defined here:https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)  # init: kaiming_uniform
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # if self.tier == 7:
        #     continue
        if self.tier == 6 or self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer1 = self._make_layer(block, 16, layers[0])
        if self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer2 = self._make_layer(block, 16, layers[1])
        if self.tier == 4 or self.tier == 3 or self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        if self.tier == 3 or self.tier == 2 or self.tier == 1:# or self.local_v2:
             self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
        if self.tier == 2 or self.tier == 1 or self.local_v2:
             self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
        if self.tier == 1:# or self.local_v2:
             self.layer6 = self._make_layer(block, 64, layers[5], stride=1)


        if self.local_loss == True:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            if self.tier == 1 or self.tier == 2:
                self.fc = nn.Linear(64 * block.expansion, num_classes)
            if self.tier == 3:
                self.fc = nn.Linear(32 * block.expansion, num_classes)
            if self.tier == 4 :
                self.fc = nn.Linear(32 * block.expansion, num_classes)
            if self.tier == 6 or self.tier == 5:
                self.fc = nn.Linear(16 * block.expansion, num_classes)
            if self.tier == 7:
                self.fc = nn.Linear(4 * block.expansion, num_classes)  

        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # print(planes)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
    
        if self.tier == 1:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
                
            x = self.layer1(x)  # B x 16 x 32 x 32
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            # print(self.layer6.state_dict()['0.conv1.weight'])
            # torch.set_grad_enabled(True)
            
            extracted_features = x
            if self.fedavg_base:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits    
    
            elif self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features               

            
        if self.tier == 2:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
            x = self.layer1(x)  # B x 16 x 32 x 32
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            x = self.layer5(x)
            extracted_features = x
    
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features              

            
        if self.tier == 3: 
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
                
            x = self.layer1(x)  # B x 16 x 32 x 32
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            
            extracted_features = x
    
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features              

            
        if self.tier == 4:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
            x = self.layer1(x)  # B x 16 x 32 x 32
            
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            extracted_features = x
    
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features              

            
                        
        if self.tier == 5:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
                
            x = self.layer1(x)  # B x 16 x 32 x 32
            
            x = self.layer2(x)  # B x 32 x 16 x 16
            extracted_features = x
            
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features             


        if self.tier == 6:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
            
            x = self.layer1(x)  # B x 16 x 32 x 32
            extracted_features = x  
    
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features            
            
        if self.tier == 7:  
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)  # B x 16 x 32 x 32
            extracted_features = x  
            
            if self.local_loss == True:
                x = self.avgpool(x)  # B x 64 x 1 x 1
                x_f = x.view(x.size(0), -1)  # B x 64
                logits = self.fc(x_f)  # B x num_classes
                return logits, extracted_features            
            return extracted_features            

            
        
class ResNet_server(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False, tier=5, local_loss=False, **kwargs):
        super(ResNet_server, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.local_loss = local_loss
        
        self.tier = tier  

        if self.tier == 7:
            self.inplanes = 16
        elif self.tier == 6 or self.tier == 5:
            self.inplanes = 64
        elif self.tier == 4:
            self.inplanes = 128 # check again
        elif self.tier == 3:
            self.inplanes = 128 # check again
        elif self.tier == 1 or self.tier == 2:
            self.inplanes = 256 # check again
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        
        
        if self.tier == 7:   
            self.layer1 = self._make_layer(block, 16, layers[0])
        if self.tier == 7 or self.tier == 6:   
            self.layer2 = self._make_layer(block, 16, layers[1])
        if self.tier == 7 or self.tier == 6 or self.tier == 5:
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        if self.tier == 7 or self.tier == 6 or self.tier == 5 or self.tier == 4:
            self.layer4 = self._make_layer(block, 32, layers[3], stride=1)
        if self.tier == 7 or self.tier == 6 or self.tier == 5 or self.tier == 4 or self.tier == 3:
            self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
        if self.tier == 7 or self.tier == 6 or self.tier == 5 or self.tier == 4 or self.tier == 3 or self.tier == 2:
            self.layer6 = self._make_layer(block, 64, layers[5], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
    
        if self.tier == 1:  
            
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x
    
        if self.tier == 2:  
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x
    
        if self.tier == 3:  
            x = self.layer5(x)
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x
            
        if self.tier == 4:  
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x
            
              
        if self.tier == 5:  
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x

        if self.tier == 6:  
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x            
            
        if self.tier == 7:  
            x = self.layer1(x)  # B x 16 x 32 x 32
            x = self.layer2(x)  # B x 32 x 16 x 16
            x = self.layer3(x)  # B x 64 x 8 x 8
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
    
            x = self.avgpool(x)  # B x 64 x 1 x 1
            x_f = x.view(x.size(0), -1)  # B x 64
            x = self.fc(x_f)  # B x num_classes
            return x
            


def resnet56_server(c, pretrained=False, path=None, tier=5, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
    

def resnet56_server_tier(c, pretrained=False, path=None, tier=5, **kwargs):
    """
    Constructs a ResNet-56 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, tier=tier, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
    
    


''' SFL model, my first resnet18 model '''

class Baseblock_SFL(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock_SFL, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output
 
    


def resnet110_SFL_local_tier_7(classes, tier=5, **kwargs):
    
    
        
    if tier == 1:
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 2:
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 3:
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 4:
        net_glob_client = ResNet(Bottleneck, [6, 6, 6, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 5:
        net_glob_client = ResNet(Bottleneck, [6, 6, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) # [1, 0, 0] and [0, 0, 0] are the same
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 6:
        net_glob_client = ResNet(Bottleneck, [6, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 7:
        net_glob_client = ResNet(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) # [1, 0, 0] and [0, 0, 0] are the same
        net_glob_server = ResNet_server(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs)

    return net_glob_client, net_glob_server

def resnet56_SFL_local_tier_7(classes, tier=5, **kwargs):

        
    if tier == 1:#1 or tier == 12:
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 2:#1:
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 0, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 3:# or tier == 2:# or tier == 4:# or tier == 2 or tier == 1:
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 0, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 4:
        net_glob_client = ResNet(Bottleneck, [3, 3, 3, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 0, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 5:# or tier == 4 or tier == 3:
        net_glob_client = ResNet(Bottleneck, [3, 3, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) # [1, 0, 0] and [0, 0, 0] are the same
        net_glob_server = ResNet_server(Bottleneck, [0, 0, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 6:
        net_glob_client = ResNet(Bottleneck, [3, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
        net_glob_server = ResNet_server(Bottleneck, [0, 3, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)
    elif tier == 7:# or tier == 6:
        net_glob_client = ResNet(Bottleneck, [0, 0, 0, 0, 0, 0], num_classes = classes, tier = tier, local_loss=True, **kwargs) # [1, 0, 0] and [0, 0, 0] are the same
        net_glob_server = ResNet_server(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs)

    return net_glob_client, net_glob_server


def resnet56_base(classes, tier=1, **kwargs):
    """ResNet-56 base model.

    Args:
      classes: Number of classes.
      tier: Tier of the model.
      **kwargs: Additional keyword arguments.
    
    Returns:
      A ResNet-56 model.
    """
    net_glob_client = ResNet(Bottleneck, [3, 3, 3, 3, 3, 3], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client

def resnet110_base(classes, tier=1, **kwargs):
    """ResNet-110 base model.

    Args:
      classes: Number of classes.
      tier: Tier of the model.
      **kwargs: Additional keyword arguments.
    
    Returns:
      A ResNet-110 model.
    """
    net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client

def resnet110_SFL_fedavg_base(classes, tier=5, **kwargs):
    net_glob_client = ResNet(Bottleneck, [6, 6, 6, 6, 6, 6], num_classes = classes, tier = tier, local_loss=True, **kwargs) 
    return net_glob_client