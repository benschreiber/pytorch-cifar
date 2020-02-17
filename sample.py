"""Test kernel filtering techniques on a pretrained CIFAR10 model."""
import copy
import os
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
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
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if device == 'cpu':
        # Trained on GPU with DataParallel, so some changes needed to load on CPU
        pyt_device = torch.device('cpu')
        checkpoint = torch.load(
            './checkpoint/ckpt.pth', map_location=pyt_device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])

    return net, testloader, classes, criterion


def test(net, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        take_n = 1
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx == take_n:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, take_n, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return acc


class IndexableNetWrapper:
    exclude = {nn.Sequential}
    no_expand = {}

    @staticmethod
    def list_network_layers(net: nn.Module):
        inw = IndexableNetWrapper
        ret = []
        children = list(net.children())
        if type(net) in inw.no_expand or (not children and type(net) not in inw.exclude):
            return [net]
        for ch in children:
            ret.extend(inw.list_network_layers(ch))
        return ret

    def __init__(self, net: nn.Module):
        self.basic_layers = self.list_network_layers(net)
        self.net = net
        self.layers_numbers = {l: i for i, l in enumerate(self.basic_layers)}

    def __getitem__(self, item: int):
        return self.basic_layers[item]

    def __iter__(self):
        return iter(self.basic_layers)

    def get_layer_idx(self, layer: nn.Module) -> Optional[int]:
        return self.layers_numbers.get(layer, None)


def prune_weight(ratio: float, layer: nn.Conv2d):
    for k, v in layer.state_dict().items():
        if 'weight' not in k:
            continue
        with torch.no_grad():
            # Set smallest weights to zero
            vf = v.flatten()
            num_to_zero = int(len(vf) * ratio)
            _, indices = torch.topk(torch.abs(vf), num_to_zero, largest=False, sorted=False)
            vf[indices] = 0.0
    return f'prune_{ratio}'


def get_approxer(net_type: type):
    from functools import partial
    if net_type == nn.Conv2d:
        return [partial(prune_weight, f) for f in (0.25,)]  # 0.5, 0.7)]
    else:
        return []


class SensitivityAnalysis:
    approximator_ct = Callable[[nn.Module], str]
    approx_key_t = Tuple[int, str]

    def __init__(self, norm):
        self.baseline_state = {}
        self.approx_state = defaultdict(lambda: defaultdict(dict))
        self.norm_func = norm

    def set_approx_state(self, approx_state: Dict[float, float], output, this_idx: int):
        baseline, base_norm, n_elem = self.baseline_state[this_idx]
        diff_norm_tensor = self.norm_func(output - baseline)  # / base_norm / n_elem
        approx_state[this_idx] = diff_norm_tensor.item()

    def set_baseline_state(self, this_layer: int, output):
        self.baseline_state[this_layer] = \
            output, self.norm_func(output), np.prod(output.shape)

    def injected_approx_forward(
            self, original_forward: Callable,
            this_idx: int, changed_idx: int, approx_name: str, x
    ):
        if this_idx == 0:
            baseline_input, _, _ = self.baseline_state[-1]
            self.set_approx_state(self.approx_state[changed_idx][approx_name], x, -1)
        output = original_forward(x)
        self.set_approx_state(self.approx_state[changed_idx][approx_name], output, this_idx)
        return output

    def injected_baseline_forward(
            self, original_forward: Callable, layer_idx: int, x
    ):
        if layer_idx == 0:
            self.set_baseline_state(-1, x)
            self.set_approx_state(self.approx_state[-1][''], x, -1)
        output = original_forward(x)
        self.set_baseline_state(layer_idx, output)
        self.set_approx_state(self.approx_state[-1][''], output, layer_idx)
        return output

    def make_injected_forward(
            self, net_w: IndexableNetWrapper,
            approx_key: Optional[approx_key_t], module: nn.Module
    ):
        original_forward = module.forward
        idx = net_w.get_layer_idx(module)
        if idx is None:
            return
        if approx_key is None:
            module.forward = partial(
                self.injected_baseline_forward, original_forward, idx
            )
        else:
            changed_idx, approx_name = approx_key
            module.forward = partial(
                self.injected_approx_forward, original_forward, idx,
                changed_idx, approx_name
            )

    def _inject_network(
            self, net_w: IndexableNetWrapper,
            approx_key: Optional[approx_key_t], need_copy: bool
    ) -> IndexableNetWrapper:
        if need_copy:
            net_w = IndexableNetWrapper(copy.deepcopy(net_w.net))
        net_w.net.apply(lambda m: self.make_injected_forward(net_w, approx_key, m))
        return net_w

    def inject_baseline(self, net: nn.Module):
        return self._inject_network(IndexableNetWrapper(net), None, True)

    def get_tests_to_run(
            self, net: nn.Module,
            layer_approx_fun: Callable[[type], Iterable[approximator_ct]]
    ) -> Iterable[Tuple[approx_key_t, IndexableNetWrapper]]:
        net_w = IndexableNetWrapper(net)
        for index, layer in enumerate(net_w):
            for approxer in layer_approx_fun(type(layer)):
                new_net_w = IndexableNetWrapper(copy.deepcopy(net))
                approx_name = approxer(new_net_w[index])
                approx_key = index, approx_name
                injected_w = self._inject_network(new_net_w, approx_key, False)
                yield approx_key, injected_w

    def compute_layerwise_diff(self) -> Dict[int, Dict[str, float]]:
        ret = {}
        for layer_id in self.approx_state.keys():
            lhs_st, rhs_sts = self.baseline_state[layer_id], self.approx_state[layer_id]
            ret[layer_id] = {
                approx: self.norm_func(lhs_st - rhs_st) for approx, rhs_st in rhs_sts.items()
            }
        return ret


def dump_approx_state(sa: SensitivityAnalysis):
    from collections import defaultdict
    import pandas as pandas
    flattened = defaultdict(dict)
    for k1, v1 in sa.approx_state.items():
        (k2, v2), = v1.items()
        for k3, v3 in v2.items():
            flattened[k1][k3] = v3
    df = pandas.DataFrame(dict(flattened))
    with open('output.html', 'w') as f:
        print(df.T.to_html(), file=f)


def main():
    net, testloader, classes, criterion = prelude()
    baseline = net
    sa = SensitivityAnalysis(lambda m: m.norm())
    baseline_w = sa.inject_baseline(baseline)

    print("Baseline: ")
    test(baseline_w.net, testloader, criterion)

    for (index, approx_name), test_net_w in sa.get_tests_to_run(baseline, get_approxer):
        print(f"Layer {index}, approximation = {approx_name}")
        test(test_net_w.net, testloader, criterion)

    dump_approx_state(sa)


if __name__ == "__main__":
    main()
