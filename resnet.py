
import torch.nn as nn
#add imports as necessary

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.functional.relu
        self.layer1 = self.new_block(block, 64, layers[0], stride=1)
        self.layer2 = self.new_block(block, 128, layers[1], stride=2)
        self.layer3 = self.new_block(block, 256, layers[2], stride=2)
        self.layer4 = self.new_block(block, 512, layers[3], stride=2)
        self.avgpool = nn.functional.avg_pool2d
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        #TODO: implement the forward function for resnet, 
        out = self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))))))
        return self.linear(out.view(out.size(0), -1))


    def new_block(self, block, planes, num_blocks, stride):
        layers = []
        #TODO: make a convolution with the above params
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

class RotNet(ResNet):
    
    def __init__(self, block, num_blocks, rot_classes=4, inference_classes=10):
        super(RotNet, self).__init__(block, num_blocks, rot_classes)
        self.linear_cifar = nn.Linear(2048, inference_classes) 
        for name,param in self.named_parameters():
            if name == "linear_cifar.weight" or name == "linear_cifar.bias":
                param.requires_grad = False
                
    def fine_tune_forward_pass(self, x):
        out = self.avgpool(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))), 4)
        return self.linear_cifar(out.view(out.size(0), -1))