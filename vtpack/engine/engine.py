# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
from . import utils
from vtpack.layers import DynamicGrainedEncoder


def get_complexity(model):
    comp_static, comp_dynamic = [], []

    def append_complexity(m):
        if isinstance(m, DynamicGrainedEncoder):
            comp = m.get_complexity()
            comp_static.append(comp["static"])
            comp_dynamic.append(comp["dynamic"])

    model.apply(append_complexity)
    comp_static = sum(comp_static).mean()
    comp_dynamic = sum(comp_dynamic).mean()
    return comp_dynamic / comp_static.clamp(min=1.0)


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode: bool = True, tb_writer=None,
                    lr_iter_inc: float = 0, dge_enable: bool = False,
                    dge_budget: float = 1.0, dge_lr_scale: float = 1.0,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for i, (samples, targets) in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss_cls = criterion(samples, outputs, targets)
            if dge_enable:
                # comp = get_complexity(model)
                comp = model.module.complexity(samples)
                comp_ratio = comp["dynamic"] / comp["static"].clamp(min=1.0)
                loss_dge = dge_lr_scale * (comp_ratio - dge_budget) ** 2
                loss = loss_dge + loss_cls
                metric_logger.update(loss_cls=loss_cls.item())
                metric_logger.update(loss_dge=loss_dge.item())
                metric_logger.update(dge_ratio=comp_ratio.item())
                metric_logger.update(comp_static=comp["static"].item())
                metric_logger.update(comp_dynamic=comp["dynamic"].item())
            else:
                loss = loss_cls

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        for param in optimizer.param_groups:
            param["lr"] += lr_iter_inc

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.write_tensorboard(tb_writer, step=i + len(data_loader) * epoch, prefix="train-iter")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.write_tensorboard(tb_writer, step=epoch, prefix="train-epoch", global_avg=True)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch: int = None, tb_writer=None, dge_enable: bool = False,
             dge_budget: float = 1.0, dge_lr_scale: float = 1.0,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for i, (images, target) in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)

        loss_cls = criterion(output, target)
        if dge_enable:
            # comp = get_complexity(model)
            comp = model.module.complexity(images)
            comp_ratio = comp["dynamic"] / comp["static"].clamp(min=1.0)
            loss_dge = dge_lr_scale * (comp_ratio - dge_budget) ** 2
            loss = loss_dge + loss_cls
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(loss_dge=loss_dge.item())
            metric_logger.update(dge_ratio=comp_ratio.item())
            metric_logger.update(comp_static=comp["static"].item())
            metric_logger.update(comp_dynamic=comp["dynamic"].item())
        else:
            loss = loss_cls

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.write_tensorboard(tb_writer, step=epoch, prefix="eval-epoch", global_avg=True)
    print("* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}"
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
