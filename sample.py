'''Test kernel filtering techniques on a pretrained CIFAR10 model.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from timeit import default_timer as timer
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

print('==> Loading checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
if device == 'cpu':
    # Trained on GPU with DataParallel, so some changes needed to load on CPU
    pyt_device = torch.device('cpu')
    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=pyt_device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
else:
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    return acc

baseline = net

print('BASELINE')
test(baseline)

kernel_labels = [k for k in net.state_dict().keys() if 'conv' in k and 'weight' in k]
#print(net.state_dict()[kernel_labels[0]])
start = timer()
for stride in (3,):
    net = copy.deepcopy(baseline) 
    for k in kernel_labels:
        print(net.state_dict()[k].size())
        with torch.no_grad():
            '''
            # Strided
            # todo compute stride
            net.state_dict()[k].flatten()[::stride][:] = 0.
            '''
            # Smallest
            num_to_zero = int(len(net.state_dict()[k].flatten()) * 0.5)
            _, indices = torch.topk(torch.abs(net.state_dict()[k].flatten()), num_to_zero, largest=False, sorted=False)
            net.state_dict()[k].flatten()[indices] = 0.0
end = timer()
#print(net.state_dict()[kernel_labels[0]])

print('FILTERED')
test(net)

test(baseline)

'''
for stride in (4,):
    scale_factor = stride / (stride - 1.)
    for k in kernel_labels:
        net.state_dict()[k] *= scale_factor

print('RESCALED')
test(net)
'''
