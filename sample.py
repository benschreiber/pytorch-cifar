"""Test kernel filtering techniques on a pretrained CIFAR10 model."""
import copy
import os

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prelude():
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    print('==> Loading checkpoint..')
    assert os.path.isdir(
        'checkpoint_saved'), 'Error: no checkpoint directory found!'
    if device == 'cpu':
        # Trained on GPU with DataParallel, so some changes needed to load on CPU
        pyt_device = torch.device('cpu')
        checkpoint = torch.load(
            './checkpoint_saved/ckpt.pth', map_location=pyt_device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load('./checkpoint_saved/ckpt.pth')
        net.load_state_dict(checkpoint['net'])

    return net, testloader, classes, criterion


def test(net, testloader, criterion):
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
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return acc


def make_filtered_net(baseline, skip_ratio):
    net = copy.deepcopy(baseline)
    for k, v in baseline.state_dict().items():
        if 'conv' not in k or 'weight' not in k:
            continue
        with torch.no_grad():
            # Set smallest weights to zero
            vf = v.flatten()
            num_to_zero = int(len(vf) * skip_ratio)
            _, indices = torch.topk(torch.abs(vf), num_to_zero, largest=False, sorted=False)
            net.state_dict()[k].flatten()[indices] = 0.0
    return net


def main():
    net, testloader, classes, criterion = prelude()
    baseline = net
    print('Baseline: ')
    test(baseline, testloader, criterion)
    for ratio in (0.25, 0.5, 0.7):
        print(f"Filter(ratio={ratio}): ")
        filtered = make_filtered_net(baseline, ratio)
        test(filtered, testloader, criterion)


if __name__ == "__main__":
    main()
