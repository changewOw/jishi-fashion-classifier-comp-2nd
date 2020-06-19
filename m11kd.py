import sys
import os
# os.system("pip install sklearn")
# os.system("pip install mmcv")
# os.system("pip install sklearn")
# os.system("pip install efficientnet_pytorch")
# os.system("pip install tqdm")
# os.system("pip install pyyaml")
# os.system("pip install addict")
# os.system("pip install yapf")
# os.system("pip install albumentations")
#
# os.makedirs("/project/train/models/", exist_ok=True)
# os.makedirs("/project/.cache/torch/checkpoints/", exist_ok=True)
# os.system("wget -P /project/.cache/torch/checkpoints/ http://10.9.0.146:8888/group1/M00/00/01/CgkA617CtkSEUFKYAAAAAGtBN_Q516.zip")
# os.system(f'rm -f /project/.cache/torch/checkpoints/*.pth')
# os.system("cd /project/.cache/torch/checkpoints/ && unzip /project/.cache/torch/checkpoints/CgkA617CtkSEUFKYAAAAAGtBN_Q516.zip")

import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
from tqdm import tqdm
import glob
import cv2
from PIL import Image
import mmcv
import sys
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
import torch.nn as nn
import gc
from collections.abc import Iterable
import onnx
from torch.nn import init

try:
    from apex import amp
except:
    print("please install apex for mixprecision training for speed")
    print()

SEED = 42
ROOT_DIR = "/home/data/14"
TARGET_DIR = "/home/data/my_train_dataset"
FOLD_FILE_SAVE = "/home/data"
LABEL_NAMES = ["ea_style", "ethnic_style", 'jp_style', "ladylike", "leisure", "maid_style", "punk"]

LABEL_NAMES_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}

HEIGHT = 224
WIDTH = 224

from efficientnet_pytorch import EfficientNet

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from albumentations import Resize, RandomCrop, CenterCrop, ImageOnlyTransform, Compose, RandomResizedCrop, \
    CoarseDropout, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, ShiftScaleRotate, ToGray

from torch.optim.optimizer import Optimizer, required
from collections import defaultdict

## optim module

from torch.utils.data.sampler import BatchSampler, Sampler
from typing import Iterator, List, Optional, Union


class BalanceClassSampler(Sampler):
    """Abstraction over data sampler.

    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(
            self, labels: List[int], mode: Union[str, int] = "downsampling"
    ):
        """
        Args:
            labels (List[int]): list of class label
                for each elem in the dataset
            mode (str): Strategy to balance classes.
                Must be one of [downsampling, upsampling]
        """
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {
            label: (labels == label).sum() for label in set(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode
                if isinstance(mode, int)
                else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_ = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 use_gc=True, gc_conv_only=False
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_gradient_threshold == 1):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_gradient_threshold == 3):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

        return loss


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


## layers module


class AdaptiveAddPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return self.mp(x) * 0.5 + self.ap(x) * 0.5


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

    def features(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        return out


class MobileNetZhou(nn.Module):
    def __init__(self, num_classes, pool_type="avg", multi_sample=False, export=False):
        super().__init__()

        self.export = export

        n_channels = 576

        self.net = MobileNetV3_Small()
        weights = torch.load("/project/train/src_repo/mbv3_small.pth.tar", map_location='cpu')['state_dict']
        # weights = torch.load("mbv3_small.pth.tar")
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        self.net.load_state_dict(weights)

        if pool_type == "avg":
            # self.avg_pool = AdaptiveConcatPool2d(1)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # self.avg_pool = AdaptiveAddPool2d(1)
            out_shape = n_channels
        elif pool_type == 'concat':
            self.avg_pool = AdaptiveConcatPool2d(1)
            out_shape = n_channels * 2
        elif pool_type == 'gem':
            self.avg_pool = GeM()
            out_shape = n_channels

        self.flat = Flatten()

        self.multi_sample = multi_sample

        output_channel = 1280
        self.classifier20 = nn.Sequential(
            nn.Linear(out_shape, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            #             nn.Conv2d(out_shape, output_channel,1,1,0,bias=True),
            # Swish_export(),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

    def forward(self, x):
        x = self.net.hs1(self.net.bn1(self.net.conv1(x)))
        x = self.net.bneck(x)

        x = self.net.hs2(self.net.bn2(self.net.conv2(x)))

        x = self.avg_pool(x)
        x = self.flat(x)

        x = self.classifier20(x)

        #         x = self.flat(x)

        output = self.classifier2(x)

        return output


from PIL import Image, ImageEnhance, ImageOps


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class AutoAug(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.autoaug = SVHNPolicy()

    def apply(self, img, **params):
        img = Image.fromarray(img)
        img = self.autoaug(img)
        return np.array(img)


class AspectResize(ImageOnlyTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if h > w:  ## h >= w
            height = self.height
            width = int(self.height * w / h)
            padh0 = padh1 = 0
            padw0 = (self.width - width) // 2
            padw1 = self.width - padw0 - width

        elif h < w:
            width = self.width
            height = int(self.width * h / w)
            padw0 = padw1 = 0
            padh0 = (self.height - height) // 2
            padh1 = self.height - padh0 - height

        else:
            height = self.height
            width = self.width
            padw0 = padw1 = padh0 = padh1 = 0
        img = cv2.resize(img, (width, height))
        img = np.pad(img, ((padh0, padh1), (padw0, padw1), (0, 0)), mode='constant', constant_values=0)
        return img


class GridMask(ImageOnlyTransform):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio

    def apply(self, img, **params):
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.

        h, w = img.shape[:2]
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)

        if isinstance(self.ratio, (tuple, list)):
            ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        else:
            ratio = self.ratio

        self.l = math.ceil(d * ratio)
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
        mask = 1 - mask

        return mask
        # img = img * mask[..., None]
        # return img


class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, states_dict, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, states_dict, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, states_dict, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, saved_model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(saved_model.state_dict(), model_path)
        self.val_score = epoch_score


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_kfold(n_split=5):
    images_list = os.listdir(TARGET_DIR)
    print(f"there are {len(images_list)} images in the folds.....")

    labels = [LABEL_NAMES_MAP[str(p.split("-")[-1][:-4])] for p in images_list]

    kf = StratifiedKFold(n_splits=n_split, random_state=SEED, shuffle=True)
    X = list(range(len(images_list)))

    for fold, (train_idxs, val_idxs) in enumerate(kf.split(X, labels)):
        train_fns = [images_list[i] for i in train_idxs]
        val_fns = [images_list[i] for i in val_idxs]

        print(len(train_fns))
        print(len(val_fns))
        break

    print('kfold split has already finished.........')
    mmcv.dump(train_fns, f"{FOLD_FILE_SAVE}/train_0.pkl")
    mmcv.dump(val_fns, f"{FOLD_FILE_SAVE}/val_0.pkl")
    return train_fns, val_fns


def prepare_data(source_dir, target_dir, clean=True):
    print("starting prepare data..")

    if clean:
        if os.name == 'nt':
            print("cant call rm in windows")
        else:
            os.system(f'rm -f {target_dir}/*.png')
            print('Done cleaning all png')

    def normalize_image(img, org_height, org_width, new_height, new_width):
        # assert img.shape[:2] == (org_height, org_width)

        image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return image_resized

    def dump_images(org_height, org_width, new_height, new_width, heights, widths, ratio):
        iid = 0
        tt = Compose([Resize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1)])

        for name in LABEL_NAMES:
            images_list = glob.glob(f"{source_dir}/{name}/*.jpg")
            print(f"{name} has {len(images_list)} images ")

            for img_name in images_list:
                img = np.array(Image.open(img_name))

                heights.add(img.shape[0])
                widths.add(img.shape[1])
                ratio.add(img.shape[0] / img.shape[1])
                label = name
                # img_norm = normalize_image(img,org_height, org_width, new_height, new_width)

                img_norm = tt(image=img)['image']

                Image.fromarray(img_norm).save(f"{target_dir}/img-{iid}-{label}.png")
                iid += 1

        return heights, widths, ratio

    heights = set()
    widths = set()
    ratio = set()

    os.makedirs(target_dir, exist_ok=True)
    org_height = None
    org_width = None
    new_height = HEIGHT
    new_width = WIDTH
    heights, widths, ratio = dump_images(org_height, org_width, new_height, new_width, heights, widths, ratio)
    print(f"heights' set is {heights}")
    print(f"widths' set is {widths}")
    print(ratio)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class JISHIDataset(Dataset):
    def __init__(self, images_list, dir, test_mode=False):
        self.images_list = images_list
        self.dir = dir
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.225, 0.225, 0.225]

        self.test_mode = test_mode
        # if not test_mode:
        #     self.grid_mask_gen = Compose([GridMask(d1=10, d2=100, rotate=7, ratio=[0.4, 0.6], p=1)])

    def get_labels(self):
        labels = list(map(lambda x: LABEL_NAMES_MAP[str(x.split("-")[-1][:-4])], self.images_list))
        return labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, ind):
        image_name = self.images_list[ind]
        image = cv2.imread(os.path.join(self.dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = LABEL_NAMES_MAP[str(image_name.split("-")[-1][:-4])]

        if not self.test_mode and config.transforms is not None:
            image = config.transforms(image=image)['image']

        if self.test_mode and config.test_transforms is not None:
            image = config.test_transforms(image=image)['image']

        # from matplotlib import pyplot as plt
        # plt.imshow(image)
        # plt.show()

        image = image / 255.
        image -= self.mean
        image /= self.std

        image_torch = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        label_torch = torch.LongTensor([label]).contiguous()
        return image_torch, label_torch


ALPHA = 0.25
GAMMA = 2
EPS = 1e-5


class FocalCrossEntropy(nn.Module):
    def forward(self, x: torch.Tensor, target: torch.Tensor):
        x = x.float()
        target = target.float()
        prob = torch.softmax(x, dim=-1)
        prob = torch.clamp(prob, EPS, 1 - EPS)
        ce = -target * torch.log_softmax(x, dim=-1)
        loss = ALPHA * torch.pow(1 - prob, GAMMA) * ce
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, ohem=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if ohem is not None:
            loss, _ = loss.topk(k=int(ohem * loss.size(0)))
        return loss.mean()


class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.9
        self.T = 20
        self.crit = DenseCrossEntropy()
    def forward(self, x, target, teacher_outputs, ohem=None):
        KD_loss = nn.KLDivLoss()(F.log_softmax(x / self.T, dim=1),
                                 F.softmax(teacher_outputs / self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                  self.crit(x, target) * (1. - self.alpha)

        return KD_loss


kd_ratio = 0.7
@torch.no_grad()
def kd_label(Y, teacher_outputs):
    # Y: [b,]  logits[b,7]
    nY = teacher_outputs * kd_ratio
    nY[range(Y.size(0)), Y] = 1.0
    return nY



@torch.no_grad()
def smooth_label(Y):
    nY = nn.functional.one_hot(Y, config.num_classes).float()
    nY += config.label_smooth / (config.num_classes - 1)
    nY[range(Y.size(0)), Y] -= config.label_smooth / (config.num_classes - 1) + config.label_smooth
    return nY


@torch.no_grad()
def mixup_data(data, targets):
    batch_size = data.size(0)
    perm = torch.randperm(batch_size).cuda()
    lam = torch.tensor(np.random.beta(config.mixup, config.mixup, batch_size), dtype=torch.float32).cuda()

    data = data * lam.view(-1, 1, 1, 1) + data[perm] * (1 - lam.view(-1, 1, 1, 1))

    targets = targets * lam.view(-1, 1) + targets[perm] * (1 - lam.view(-1, 1))

    return data, targets


def rand_bbox(lam):
    H, W = HEIGHT, WIDTH
    cut_rat = torch.sqrt(1 - lam)
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()

    cx = torch.randint(high=W, size=[1]).cuda()
    cy = torch.randint(high=H, size=[1]).cuda()

    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def cutmix_data(data, targets):
    targets = targets.float()
    batch_size = data.size(0)
    perm = torch.randperm(batch_size).cuda()
    lam = torch.tensor(np.random.beta(config.cutmix, config.cutmix), dtype=torch.float32).cuda()  # max(lam, 1-lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data[perm, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2.float() - bbx1.float()) * (bby2.float() - bby1.float()) / (HEIGHT * WIDTH))
    targets = targets * lam + targets[perm] * (1 - lam)

    return data, targets

    ##


def train_one(dataloader, model, device, optimizer, scheduler=None, mixup=False, cutmix=False, ema_model=None, need_apex=True, ohem=None):
    model, model_teacher = model  ## split

    model.train()

    for bi, data in enumerate(dataloader):
        images, labels, g_mask = data[0], data[1].squeeze(), data[2]

        images = images.to(device)


        if config.label_smooth is not None:
            labels_onehot = smooth_label(labels)
        else:
            labels_onehot = nn.functional.one_hot(labels, config.num_classes)

        optimizer.zero_grad()

        labels_onehot = labels_onehot.to(device)


        # if mixup and cutmix:  # todo mixup && cutmix
        #     if np.random.uniform() < 0.5:
        #         images, labels_onehot = mixup_data(images, labels_onehot)
        #     else:
        #         images, labels_onehot = cutmix_data(images, labels_onehot)
        # else:

        if mixup and config.mixup_prob > 0: # todo mixup cutmix
            if np.random.uniform() < config.mixup_prob:
                images, labels_onehot = mixup_data(images, labels_onehot)


        r = np.random.uniform()
        if cutmix and config.cutmix_prob > 0:  # 0.5
            if r < config.cutmix_prob:
                images, labels_onehot = cutmix_data(images, labels_onehot)

        # if config.cutmix_prob <= r :
        #     images = g_mask[:, None, ...].cuda() * images


        outputs = model(images)
        with torch.no_grad():
            outputs_teacher = model_teacher(images)
        loss = config.loss_fn(outputs, labels_onehot, outputs_teacher, ohem)

        if need_apex and config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if ema_model is not None:
            accumulate(ema_model, model, decay=0.9)

def val_one(dataloader, model, device):
    model.eval()
    # losses = AverageMeter()
    accs = []
    # p = []
    # l = []
    with torch.no_grad():
        for bi, data in enumerate(dataloader):
            images, labels = data[0], data[1].squeeze()

            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            acc = (labels.numpy() == (probs.argmax(-1))).astype(np.uint8)
            accs.extend(acc)
            # p.append(probs.argmax(-1))
            # l.append(labels.numpy())

    accs_avg = np.array(accs).mean()
    # recall = sklearn.metrics.recall_score(l, p, average='macro')
    print(f"Valid dataset Accuracy is {accs_avg}")
    # print(f"Valid dataset losses is {losses.avg}")
    # print(f"Valid dataset recall is {recall}")
    print()
    return accs_avg, None


def accumulate(model1, model2, decay=0.99):
    par1 = model1.state_dict()
    par2 = model2.state_dict()

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))


def run(train_fns, val_fns):
    print(f"Training images has {len(train_fns)} , Valid images has {len(val_fns)},  ")

    ## train stage1
    train_dataset = JISHIDataset(train_fns, TARGET_DIR, test_mode=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode='upsampling'),
        # shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    val_dataset = JISHIDataset(val_fns, TARGET_DIR, test_mode=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    del train_dataset, val_dataset
    gc.collect()

    device = "cuda"

    with torch.no_grad():
        model_teacher = EfficientnetZhou()
        model_teacher.eval()
        model_teacher.load_state_dict(torch.load("/project/train/pre-trained-models/v1p/model_0.pth"))
        model_teacher.to(device)
        print("teacher model has been loading...................")



    model = MobileNetZhou(num_classes=config.num_classes, pool_type='avg', multi_sample=False, export=False)
    model.to(device)
    optimizer = RAdam(model.parameters(), lr=1e-3)
    # optimizer = Ranger(model.parameters(), lr=1e-2)
    # optimizer = Over9000(model.parameters(), lr=1e-2)

    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0,
                                                num_training_steps=config.epochs * len(train_loader))

    print("!!!!!!!!!!!!!!!!!!!!!!!staring training !!!!!!!!!!!!!!!!!!!!!!!")
    print()
    es = EarlyStopping(patience=4, mode='max', delta=1e-5)
    # ohems = [None,] * 5 + list(reversed(list(np.linspace(0.2, 1.0, 12)))) + [0.2] * 3

    for epoch in range(config.epochs):
        train_one(train_loader, [model, model_teacher], device, optimizer, scheduler, mixup=config.MIXUP, cutmix=config.CUTMIX, ohem=None)
        scores, losses = val_one(val_loader, model, device)
        es(scores, model, f"/project/train/models/v1/model_{config.fold}.pth")

    torch.save(model.state_dict(), f"/project/train/models/v1/model_{config.fold}.pth")

    ## train stage2
    #

    del train_loader, val_loader, scheduler
    gc.collect()

    train_dataset = JISHIDataset(train_fns + val_fns, TARGET_DIR, test_mode=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    del train_dataset
    gc.collect()
    model.load_state_dict(torch.load(f"/project/train/models/v1/model_{config.fold}.pth"))
    model.to(device)

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-6

    n_splits = 4
    for epoch in range(n_splits):
        train_one(train_loader, [model, model_teacher], device, optimizer, None, mixup=False, cutmix=False, ohem=None)
        torch.save(model.state_dict(), f"/project/train/models/v1_ema/ema_model_{config.fold}_{epoch}.pth")

    state_dict = torch.load(f"/project/train/models/v1_ema/ema_model_{config.fold}_0.pth")
    for k, v in state_dict.items():
        state_dict[k] = v / n_splits
    for sss in [torch.load(f"/project/train/models/v1_ema/ema_model_{config.fold}_{epoch}.pth") for epoch in
                range(1, n_splits)]:
        for k, v in sss.items():
            state_dict[k] += v / n_splits

    torch.save(state_dict, f"/project/train/models/v1_ema/ema_model_{config.fold}.pth")

    ###############################################################################################################
    ## create onnx
    del model
    gc.collect()
    torch.cuda.empty_cache()

    dummy = torch.randn((1, 3, HEIGHT, WIDTH))
    dummy_tta = torch.randn(2, 3, HEIGHT, WIDTH)

    os.makedirs("/project/train/models/deploy/", exist_ok=True)

    if config.apex:
        with amp.disable_casts():
            ## stage 1
            model = MobileNetZhou(num_classes=config.num_classes, pool_type='avg', multi_sample=False, export=True)
            model.load_state_dict(torch.load(f"/project/train/models/v1/model_{config.fold}.pth"))
            model.eval().cpu()
            model = torch.quantization.fuse_modules(model, config.merge_list)
            torch.jit.trace(model, dummy).save("/project/train/models/deploy/deploy_stage1.pt")
            torch.onnx.export(model, dummy, f"/project/train/models/deploy/deploy_stage1.onnx", verbose=False)
            # torch.onnx.export(model, dummy_tta, f"/project/train/models/v1/model_{config.fold}_tta.onnx", verbose=False)

            # ## stage 2
            # model = GhostNetZhou(num_classes=config.num_classes, pool_type='avg', multi_sample=False, export=True)
            # model.load_state_dict(torch.load(f"/project/train/models/v1_ema/ema_model_{config.fold}.pth"))
            # model.eval().cpu()
            # torch.jit.trace(model, dummy).save("/project/train/models/deploy/deploy_stage2.pt")
            # torch.onnx.export(model, dummy, f"/project/train/models/deploy/deploy_stage2.onnx", verbose=False)
            # torch.onnx.export(model, dummy_tta, f"/project/train/models/v1/model_{config.fold}_tta.onnx", verbose=False)

            # print(f"{a}")
    else:
        ## stage 1
        model = MobileNetZhou(num_classes=config.num_classes, pool_type='avg', multi_sample=False, export=True)
        model.load_state_dict(torch.load(f"/project/train/models/v1/model_{config.fold}.pth"))
        model.eval().cpu()
        model = torch.quantization.fuse_modules(model, config.merge_list)
        torch.jit.trace(model, dummy).save("/project/train/models/deploy/deploy_stage1.pt")
        torch.onnx.export(model, dummy, f"/project/train/models/deploy/deploy_stage1.onnx", verbose=False)
        ## stage 2
        model = MobileNetZhou(num_classes=config.num_classes, pool_type='avg', multi_sample=False, export=True)
        model.load_state_dict(torch.load(f"/project/train/models/v1_ema/ema_model_{config.fold}.pth"))
        model.eval().cpu()
        model = torch.quantization.fuse_modules(model, config.merge_list)
        torch.jit.trace(model, dummy).save("/project/train/models/deploy/deploy_stage2.pt")
        torch.onnx.export(model, dummy, f"/project/train/models/deploy/deploy_stage2.onnx", verbose=False)


class Config():
    def __init__(self):
        self.prepare_data = True
        self.make_kfold = True
        self.run = True

        self.fold = 0
        self.num_classes = 7

        ## train configuration
        self.epochs = 20
        self.batch_size = 256
        self.num_workers = 0

        ## stage2 train configuration

        self.transforms = Compose([
            # AspectResize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
            Resize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
            RandomCrop(height=HEIGHT, width=WIDTH, p=1),
            # AspectResize(HEIGHT, WIDTH, p=1),
            # Resize(HEIGHT, WIDTH, p=1),
            HorizontalFlip(p=0.5),
            # AutoAug(p=1),
            # ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=5, p=1),
            #             HueSaturationValue(p=0.33),
            #             RandomBrightnessContrast(p=0.5),
            # GridMask(d1=10, d2=100, rotate=7, ratio=[0.4, 0.6], p=0.5)
        ])
        self.test_transforms = Compose([
            Resize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
            # AspectResize(int(HEIGHT * (256 / 224)), int(WIDTH * (256 / 224)), p=1),
            CenterCrop(height=HEIGHT, width=WIDTH, p=1),
            # AspectResize(HEIGHT, WIDTH, p=1),
            # Resize(HEIGHT, WIDTH, p=1),
        ])


        self.loss_fn = KDLoss()
        # self.loss_fn = DenseCrossEntropy()
        # self.loss_fn_metric = ArcFaceLoss()
        # self.loss_fn = FocalCrossEntropy()

        self.apex = False
        self.label_smooth = None
        self.mixup = 1.0
        self.mixup_prob = 0.5
        self.MIXUP = False

        self.cutmix = 1.0
        self.cutmix_prob = 0.5
        self.CUTMIX = True  ##

        self.merge_list = [
            ["net.conv1", "net.bn1"],
            ["net.bneck.0.se.se.1", "net.bneck.0.se.se.2", "net.bneck.0.se.se.3"],
            ["net.bneck.0.se.se.4", "net.bneck.0.se.se.5"],

            ["net.bneck.0.conv1", "net.bneck.0.bn1", "net.bneck.0.nolinear1"],
            ["net.bneck.0.conv2", "net.bneck.0.bn2", "net.bneck.0.nolinear2"],
            ["net.bneck.0.conv3", "net.bneck.0.bn3"],

            ## 1for
            ["net.bneck.1.conv1", "net.bneck.1.bn1", "net.bneck.1.nolinear1"],
            ["net.bneck.1.conv2", "net.bneck.1.bn2", "net.bneck.1.nolinear2"],
            ["net.bneck.1.conv3", "net.bneck.1.bn3"],

            ## 2for
            ["net.bneck.2.conv1", "net.bneck.2.bn1", "net.bneck.2.nolinear1"],
            ["net.bneck.2.conv2", "net.bneck.2.bn2", "net.bneck.2.nolinear2"],
            ["net.bneck.2.conv3", "net.bneck.2.bn3"],

            ## 3for
            ["net.bneck.3.se.se.1", "net.bneck.3.se.se.2", "net.bneck.3.se.se.3"],
            ["net.bneck.3.se.se.4", "net.bneck.3.se.se.5"],

            ["net.bneck.3.conv1", "net.bneck.3.bn1"],
            ["net.bneck.3.conv2", "net.bneck.3.bn2"],
            ["net.bneck.3.conv3", "net.bneck.3.bn3"],

            ## 4for
            ["net.bneck.4.se.se.1", "net.bneck.4.se.se.2", "net.bneck.4.se.se.3"],
            ["net.bneck.4.se.se.4", "net.bneck.4.se.se.5"],

            ["net.bneck.4.conv1", "net.bneck.4.bn1"],
            ["net.bneck.4.conv2", "net.bneck.4.bn2"],
            ["net.bneck.4.conv3", "net.bneck.4.bn3"],

            ## 5for
            ["net.bneck.5.se.se.1", "net.bneck.5.se.se.2", "net.bneck.5.se.se.3"],
            ["net.bneck.5.se.se.4", "net.bneck.5.se.se.5"],

            ["net.bneck.5.conv1", "net.bneck.5.bn1"],
            ["net.bneck.5.conv2", "net.bneck.5.bn2"],
            ["net.bneck.5.conv3", "net.bneck.5.bn3"],

            ## 6for
            ["net.bneck.6.se.se.1", "net.bneck.6.se.se.2", "net.bneck.6.se.se.3"],
            ["net.bneck.6.se.se.4", "net.bneck.6.se.se.5"],

            ["net.bneck.6.conv1", "net.bneck.6.bn1"],
            ["net.bneck.6.conv2", "net.bneck.6.bn2"],
            ["net.bneck.6.conv3", "net.bneck.6.bn3"],

            ["net.bneck.6.shortcut.0", "net.bneck.6.shortcut.1"],

            ## 7for
            ["net.bneck.7.se.se.1", "net.bneck.7.se.se.2", "net.bneck.7.se.se.3"],
            ["net.bneck.7.se.se.4", "net.bneck.7.se.se.5"],

            ["net.bneck.7.conv1", "net.bneck.7.bn1"],
            ["net.bneck.7.conv2", "net.bneck.7.bn2"],
            ["net.bneck.7.conv3", "net.bneck.7.bn3"],

            ## 8for
            ["net.bneck.8.se.se.1", "net.bneck.8.se.se.2", "net.bneck.8.se.se.3"],
            ["net.bneck.8.se.se.4", "net.bneck.8.se.se.5"],

            ["net.bneck.8.conv1", "net.bneck.8.bn1"],
            ["net.bneck.8.conv2", "net.bneck.8.bn2"],
            ["net.bneck.8.conv3", "net.bneck.8.bn3"],

            ## 9for
            ["net.bneck.9.se.se.1", "net.bneck.9.se.se.2", "net.bneck.9.se.se.3"],
            ["net.bneck.9.se.se.4", "net.bneck.9.se.se.5"],

            ["net.bneck.9.conv1", "net.bneck.9.bn1"],
            ["net.bneck.9.conv2", "net.bneck.9.bn2"],
            ["net.bneck.9.conv3", "net.bneck.9.bn3"],

            ## 10
            ["net.bneck.10.se.se.1", "net.bneck.10.se.se.2", "net.bneck.10.se.se.3"],
            ["net.bneck.10.se.se.4", "net.bneck.10.se.se.5"],

            ["net.bneck.10.conv1", "net.bneck.10.bn1"],
            ["net.bneck.10.conv2", "net.bneck.10.bn2"],
            ["net.bneck.10.conv3", "net.bneck.10.bn3"],

            ##
            ["net.conv2", "net.bn2"],

        ]


if __name__ == '__main__':

    print("1123")

    config = Config()
    seed_torch(SEED)

    if config.prepare_data:
        prepare_data(ROOT_DIR, TARGET_DIR, clean=True)

    if config.make_kfold:
        train_fns, val_fns = make_kfold(n_split=10)
    else:
        train_fns, val_fns = mmcv.load(f"{FOLD_FILE_SAVE}/train_0.pkl"), mmcv.load(f"{FOLD_FILE_SAVE}/val_0.pkl")


    os.makedirs("/project/train/models/v1/", exist_ok=True)
    os.makedirs("/project/train/models/v1_ema/", exist_ok=True)

    if config.run:
        run(train_fns, val_fns)

"""

{0.971875, 
1.4952978056426331,
 1.4008620689655173, 
 1.25,
  1.4974093264248705, 
  1.5,
   2.458422174840085,
   1.4986376021798364, 
   1.644,
    1.43125,
     1.4038876889848813,
      1.436, 
      1.0, 
      1.2771855010660982, 
      1.282857142857143, 
      2.2115384615384617,
       2.9873417721518987, 
       1.2933333333333332, 
       0.9226666666666666, 
       1.4359375, 
       1.202469135802469, 
       2.4248704663212437, 
       2.8466257668711656, 
       1.3328125, 
       1.216, 
       1.275925925925926, 
       1.0177777777777777, 
       1.0546666666666666, 0.9919484702093397, 1.5573333333333332, 1.955032119914347, 0.9875, 1.1546666666666667, 1.0042194092827004, 1.987987987987988, 1.1866666666666668, 1.3333333333333333, 1.499267935578331, 1.4981273408239701, 1.499, 2.13953488372093, 2.3794871794871795, 1.3987068965517242, 1.3371150729335495}
"""














