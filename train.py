# -*- coding: utf-8 -*-
import os
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter

from autoaug import CIFAR10Policy, Cutout
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from models.resnet32_cifar_group import ResNet32Model
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn



# ---------------------- utils ----------------------
class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = self.sum = self.count = self.avg = 0.

    def update(self, v, n=1):
        if torch.is_tensor(v):
            v = v.item()
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})'


class ARBLoss(torch.nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer('cls_num_list', cls_num_list)

    def forward(self, logits, targets):
        n_i = self.cls_num_list[targets]
        n_k = self.cls_num_list.unsqueeze(0)
        coeff = n_k / n_i.unsqueeze(1)
        logits_exp = torch.exp(logits)
        denom = (coeff * logits_exp).sum(dim=1)
        logits_true = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -logits_true + torch.log(denom + 1e-8)
        return loss.mean()

class IGRReweightCELoss(torch.nn.Module):

    def __init__(self, cls_num_list, alpha=1.0, prior_type='inv_freq', max_w=10.0, eps=1e-8):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer('cls_num_list', cls_num_list)


        freq = cls_num_list / cls_num_list.sum()
        if prior_type == 'uniform':
            prior_w = torch.ones_like(freq)
        elif prior_type == 'inv_sqrt_freq':
            prior_w = 1.0 / torch.sqrt(freq + eps)
        else:
            prior_w = 1.0 / (freq + eps)

        prior_w = prior_w / prior_w.mean()
        self.register_buffer('prior_w', prior_w)

        self.alpha = alpha
        self.max_w = max_w
        self.eps = eps


        self.register_buffer('cls_batch_cnt', torch.zeros_like(cls_num_list))

        self.register_buffer('macro_weight', torch.ones_like(cls_num_list))

        self.macro_enabled = False

    @torch.no_grad()
    def compute_macro_weight_from_batch_count(self, gamma: float = 0.5, min_count: float = 1.0):

        B = self.cls_batch_cnt.float()
        B_clamped = torch.clamp(B, min=min_count)

        beta = B_clamped.pow(-gamma)
        mean_beta = beta.mean()
        if mean_beta > 0:
            beta = beta / mean_beta

        self.macro_weight.copy_(beta)
        self.macro_enabled = True

    def forward(self, logits, targets):
        """
        logits: [B, C]
        targets: [B]
        """
        ce = F.cross_entropy(logits, targets, reduction='none')  # per-sample CE
        B, C = logits.size()
        device = logits.device

        ell_c = torch.zeros(C, device=device)
        present_mask = torch.zeros(C, dtype=torch.bool, device=device)
        for c in range(C):
            idx = (targets == c)
            if idx.any():
                ell_c[c] = ce[idx].mean()
                present_mask[c] = True

        if not present_mask.any():
            return ce.mean()


        self.cls_batch_cnt[present_mask] += 1


        ell_target = ell_c[present_mask].mean()

        w0 = self.prior_w.to(device)
        w = w0.clone()

        denom = ell_c * ell_c + self.alpha + self.eps
        numer = ell_target * ell_c + self.alpha * w0

        w[present_mask] = numer[present_mask] / denom[present_mask]



        if self.macro_enabled:
            w = w * self.macro_weight.to(device)


        w = torch.clamp(w, min=0.0, max=self.max_w)

        sample_w = w[targets]  # [B]
        loss = (sample_w * ce).mean()
        return loss


@torch.no_grad()
def accuracy(logits, target, topk=(1,)):
    maxk = max(topk)
    B = target.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    corr = pred.eq(target.view(1, -1).expand_as(pred))
    return [corr[:k].reshape(-1).float().sum() * 100. / B for k in topk]



def _normalized_entropy(counts):

    n = float(sum(counts))
    p = [c / n for c in counts if c > 0]
    if not p:
        return 1.0
    H = -sum(pi * math.log(pi) for pi in p)
    C = len(counts)
    return H / math.log(C)


def _mittag_leffler_Ea_neg(z, a):

    if z < 1.0:
        s = 1.0
        for k in range(1, 32):
            s += ((-z) ** k) / math.gamma(a * k + 1.0)
        return max(s, 0.0)
    else:
        return max(1.0 / (z * math.gamma(1.0 - a)), 0.0)


class MiLeLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, total_steps, counts=None,
                 warmup_steps=0, switch_steps=None, last_epoch=-1, eps=1e-6):
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.switch_steps = None if (switch_steps is None or int(switch_steps) <= 0) else int(switch_steps)
        self.eps = float(eps)
        H_norm = 1.0 if counts is None else _normalized_entropy(counts)
        self.alpha = 0.25 + 0.75 * H_norm
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]

        t = max(step - self.warmup_steps, 0)
        T = max(self.total_steps - self.warmup_steps, 1)

        if self.switch_steps is None or self.switch_steps >= T:
            s = min(t / T, 1 - self.eps)
            z = s / (1.0 - s)
            f = _mittag_leffler_Ea_neg(z, self.alpha)
            return [base_lr * f for base_lr in self.base_lrs]

        if t < self.switch_steps:
            p = t / max(self.switch_steps, 1)
            z1 = (1.0 - self.eps) * p
            f = _mittag_leffler_Ea_neg(z1, self.alpha)
        else:
            t2 = t - self.switch_steps
            T2 = max(T - self.switch_steps, 1)
            s2 = min(t2 / T2, 1 - self.eps)
            z2 = 1.0 + s2 / (1.0 - s2 + self.eps)
            f = max(1.0 / (z2 * math.gamma(1.0 - self.alpha)), 0.0)

        return [base_lr * f for base_lr in self.base_lrs]


# ---------------------- train & validate ----------------------
def train(loader, model, ce_loss_fn, igr_loss_fn, optimizer, lr_scheduler, epoch, args, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc@1', ':6.2f')

    model.train()
    tic = time.time()
    for i, (views, tgt) in enumerate(loader):

        x = torch.cat(views, 0).cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)

        outputs = model(x)
        logits_all = outputs[1]

        logits = logits_all.chunk(3, 0)[0]


        if args.loss_mode == 'all_reweighting':
            loss = igr_loss_fn(logits, tgt)
        elif args.loss_mode == 'ce_then_reweighting':
            if epoch < args.switch_epoch:
                loss = ce_loss_fn(logits, tgt)
            else:
                loss = igr_loss_fn(logits, tgt)
        else:

            loss = ce_loss_fn(logits, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        acc1 = accuracy(logits, tgt)[0]
        loss_meter.update(loss.item(), tgt.size(0))
        acc_meter.update(acc1.item(), tgt.size(0))
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            print(f'Ep[{epoch:03d}][{i:03d}/{len(loader):03d}] '
                  f'LR {optimizer.param_groups[0]["lr"]:.4f} '
                  f'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  f'Loss {loss_meter.val:.4f}({loss_meter.avg:.4f}) '
                  f'Acc@1 {acc_meter.val:.2f}({acc_meter.avg:.2f})')

        step = epoch * len(loader) + i
        writer.add_scalar('loss/train', loss_meter.val, step)
        writer.add_scalar('acc/train', acc_meter.val, step)


@torch.no_grad()
def validate(loader, model, epoch, writer=None, flag='val'):
    model.eval()
    acc_meter = AverageMeter('Acc@1', ':6.2f')
    for imgs, tgt in loader:
        imgs, tgt = imgs.cuda(), tgt.cuda()
        logits = model(imgs)[1]
        acc_meter.update(accuracy(logits, tgt)[0], tgt.size(0))
    print(f'{flag} Acc@1 {acc_meter.avg:.2f}')
    if writer:
        writer.add_scalar(f'acc/{flag}', acc_meter.avg, epoch)
    return acc_meter.avg


# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser('SGD + CE / Reweighting')
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--imb_type', default='exp')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--alpha', type=float, default=0,
                        help='IGR Tikhonov regularization strength')
    parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'inv_freq', 'inv_sqrt_freq'],
                        help='prior type w_c^(0) for IGR')


    parser.add_argument('--loss_mode', type=str, default='ce_then_reweighting',
                        choices=['all_reweighting', 'ce_then_reweighting'])
    parser.add_argument('--switch_epoch', type=int, default=160)
    parser.add_argument('--macro_switch_epoch', type=int, default=160)
    parser.add_argument('--macro_gamma', type=float, default=1)


    parser.add_argument('--lr_switch_epoch', type=int, default=160)


    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic, cudnn.benchmark = True, False

    tf_train = [
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(1, 16),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    ]
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    if args.dataset == 'cifar10':
        train_set = IMBALANCECIFAR10('/data', args.imb_type, args.imb_factor,
                                     train=True, download=True, transform=tf_train)
        val_set = datasets.CIFAR10('/data', train=False, download=True, transform=tf_val)
    else:
        train_set = IMBALANCECIFAR100('/data', args.imb_type, args.imb_factor,
                                      train=True, download=True, transform=tf_train)
        val_set = datasets.CIFAR100('/data', train=False, download=True, transform=tf_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    cls_num_list = train_set.get_cls_num_list()
    num_classes = len(cls_num_list)
    model = ResNet32Model(num_classes, use_norm=True, classifier=True).cuda()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )


    ce_loss_fn = nn.CrossEntropyLoss().cuda()

    igr_loss_fn = IGRReweightCELoss(
        cls_num_list=cls_num_list,
        alpha=args.alpha,
        prior_type=args.prior,
    ).cuda()

    writer = SummaryWriter('./logs_igr_ce')

    best_acc, start_epoch = 0.0, 0
    acc_history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f'=> Resume from epoch {start_epoch}, best {best_acc:.2f}%')


    if args.loss_mode == 'ce_then_reweighting':
        assert args.macro_switch_epoch >= args.switch_epoch, \
            "macro_switch_epoch  ≥ switch_epoch"


    iters_per_epoch = len(train_loader)
    total_steps = args.epochs * iters_per_epoch
    warmup_steps = max(int(args.warmup_epochs * iters_per_epoch), 0)

    lr_switch_epoch = getattr(args, 'lr_switch_epoch', -1)
    if lr_switch_epoch is not None and lr_switch_epoch >= 0:
        switch_global_steps = int(lr_switch_epoch * iters_per_epoch)
        switch_steps = max(switch_global_steps - warmup_steps, 0)
    else:
        switch_steps = None


    last_epoch_steps = max(start_epoch * iters_per_epoch - 1, -1)


    for g in optimizer.param_groups:
        g.setdefault('initial_lr', args.lr)

    lr_scheduler = MiLeLR(
        optimizer,
        total_steps=total_steps,
        counts=cls_num_list,
        warmup_steps=warmup_steps,
        switch_steps=switch_steps,
        last_epoch=last_epoch_steps,
    )

    for ep in range(start_epoch, args.epochs):

        if ep >= getattr(args, 'macro_switch_epoch', -1):
            if hasattr(igr_loss_fn, 'compute_macro_weight_from_batch_count'):
                igr_loss_fn.compute_macro_weight_from_batch_count(
                    gamma=getattr(args, 'macro_gamma', 0.5)
                )
                if ep == args.macro_switch_epoch:
                    print(f'=> Enable macro batch-count reweight from epoch {ep}')

        train(train_loader, model, ce_loss_fn, igr_loss_fn, optimizer, lr_scheduler, ep, args, writer)
        val_acc = validate(val_loader, model, ep, writer)

        best_acc = max(best_acc, val_acc)
        print(f'Epoch {ep:03d}  Val {val_acc:.2f}%   Best-so-far {best_acc:.2f}%')

        acc_history.append(float(val_acc))

        torch.save({
            'epoch': ep + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, 'checkpoint_ce_only.pth.tar')

    print(f'Finished. Best Val Acc@1 = {best_acc:.2f}%')



if __name__ == '__main__':
    main()
