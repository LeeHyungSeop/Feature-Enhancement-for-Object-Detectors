"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 
import torch.nn as nn
import torch.nn.functional as F

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

torch.autograd.set_detect_anomaly(True)

def js_divergence(p, q):
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    """
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + F.kl_div(q.log(), m, reduction='batchmean'))

def compute_js_distance(tensor1, tensor2):
    """
    Compute the Jensen-Shannon distance between two tensors.
    """
    # Reshape tensors to 2D
    tensor1 = tensor1.view(tensor1.size(0), -1)
    tensor2 = tensor2.view(tensor2.size(0), -1)

    # Convert tensors to probability distributions
    p = F.softmax(tensor1, dim=1)
    q = F.softmax(tensor2, dim=1)

    # Compute the JS divergence
    js_div = js_divergence(p, q)

    # Convert divergence to distance
    js_distance = torch.sqrt(js_div)
    return js_distance


def train_one_epoch_twoBackward(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            
            # 2024.08.01 @hslee KL-Div (forward-backward, forward-backward, update)
            
            optimizer.zero_grad()
            # 0.5, 0.7, 0.9
            alpha = 0.5 # wneck : weight to alpha, w/o neck : weight to (1 - alpha)
            T = 4.0
             
            wNeck = True
            outputs_w_neck = model(samples, targets, wNeck=wNeck)
            backbone_outs = outputs_w_neck.pop('backbone_outs') 
            neck_outs = outputs_w_neck.pop('neck_outs')
            
            loss_w_neck_dict = criterion(outputs_w_neck, targets)
            loss_w_neck = sum(loss_w_neck_dict.values())
            loss_w_neck = alpha * loss_w_neck

            loss_w_neck.backward(retain_graph=True)
            

            wNeck = False
            outputs_wo_neck = model(samples, targets, wNeck=wNeck)
            loss_wo_neck_dict = criterion(outputs_wo_neck, targets)
            loss_wo_neck = sum(loss_wo_neck_dict.values())
            
            loss_nb_kd = 0
            num_scales = len(neck_outs)
            kl_div = nn.KLDivLoss(reduction='batchmean')
            
            # KL-Div [bs, c*h*w]
            for i in range(num_scales):
                # make [bs, c, h, w] -> [bs, c*h*w] -> make probability distribution by softmax
                student = backbone_outs[i].reshape(backbone_outs[i].shape[0], -1)
                student = F.log_softmax(student / T, dim=1)
                teacher = neck_outs[i].reshape(neck_outs[i].shape[0], -1)
                teacher = F.softmax(teacher / T, dim=1)
                
                # N -> B KL-Div
                loss_nb_kd += kl_div(student, teacher) * (T * T)
            loss_nb_kd /= num_scales
            
            loss_wo_neck = (1 - alpha) * loss_wo_neck + loss_nb_kd.item()
            try:
                loss_wo_neck.backward()
            except RuntimeError as err:
                print(err)                    
            
            optimizer.step()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_w_neck_reduced = reduce_dict(loss_w_neck_dict)
        loss_dict_wo_neck_reduced = reduce_dict(loss_wo_neck_dict)
        
        loss_value_w_neck = sum(loss_dict_w_neck_reduced.values())
        loss_value_wo_neck = sum(loss_dict_wo_neck_reduced.values())
        
        loss_value = alpha * loss_value_w_neck + (1 - alpha) * loss_value_wo_neck + loss_nb_kd

        if not math.isfinite(loss_value_w_neck):
            print("Loss is {}, stopping training".format(loss_value_w_neck))
            print(loss_dict_w_neck_reduced)
            sys.exit(1)
        if not math.isfinite(loss_value_wo_neck):
            print("Loss is {}, stopping training".format(loss_value_wo_neck))
            print(loss_dict_wo_neck_reduced)
            sys.exit(1)

        
        metric_logger.update(loss=loss_value)
        # 2024.08.03 @hslee No KD Loss
        metric_logger.update(KL=loss_nb_kd)
        metric_logger.update(loss_w_neck=loss_value_w_neck, **loss_dict_w_neck_reduced)
        metric_logger.update(loss_wo_neck=loss_value_wo_neck, **loss_dict_wo_neck_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_oneBackward(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            
            optimizer.zero_grad()

            alpha = 0.5 # wneck : weight to alpha, w/o neck : weight to (1 - alpha)
            T = 4.0
             
            wNeck = True
            outputs_w_neck = model(samples, targets, wNeck=wNeck)
            loss_w_neck_dict = criterion(outputs_w_neck, targets)
            loss_w_neck = sum(loss_w_neck_dict.values())
            
            # wNeck = False
            # outputs_wo_neck = model(samples, targets, wNeck=wNeck)
            # loss_wo_neck_dict = criterion(outputs_wo_neck, targets)
            # loss_wo_neck = sum(loss_wo_neck_dict.values())
            
            # num_scales = len(neck_outs)
            # kl_div = nn.KLDivLoss(reduction='batchmean')
            
            # # KL-Div [bs, c*h*w]
            # student = student.view(student.shape[0], -1)
            # student = F.log_softmax(student / T, dim=1)
            # teacher = teacher.view(teacher.shape[0], -1)
            # teacher = F.softmax(teacher / T, dim=1)
            # loss_nb_kd = kl_div(student, teacher) * (T * T)
            
            # MSE Loss
            # loss_nb_kd = F.mse_loss(student, teacher.clone().detach())
            
            
            # for i in range(num_scales):
            #     # make [bs, c, h, w] -> [bs, c*h*w] -> make probability distribution by softmax
            #     student = backbone_outs[i].reshape(backbone_outs[i].shape[0], -1).clone()
            #     student = F.log_softmax(student / T, dim=1)
            #     teacher = neck_outs[i].reshape(neck_outs[i].shape[0], -1).clone().detach()
            #     teacher = F.softmax(teacher / T, dim=1)
            #     # N -> B KL-Div
            #     loss_nb_kd += kl_div(student, teacher) * (T * T)
            # loss_nb_kd /= num_scales
                
            
            loss = loss_w_neck
            loss.backward()
            optimizer.step()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_w_neck_reduced = reduce_dict(loss_w_neck_dict)
        # loss_dict_wo_neck_reduced = reduce_dict(loss_wo_neck_dict)
        
        loss_value_w_neck = sum(loss_dict_w_neck_reduced.values())
        # loss_value_wo_neck = sum(loss_dict_wo_neck_reduced.values())
        # loss_value = alpha * loss_value_w_neck + (1 - alpha) * loss_value_wo_neck + loss_nb_kd
        loss_value = loss

        if not math.isfinite(loss_value_w_neck):
            print("Loss is {}, stopping training".format(loss_value_w_neck))
            print(loss_dict_w_neck_reduced)
            sys.exit(1)
        # if not math.isfinite(loss_value_wo_neck):
        #     print("Loss is {}, stopping training".format(loss_value_wo_neck))
        #     print(loss_dict_wo_neck_reduced)
        #     sys.exit(1)

        
        metric_logger.update(loss=loss_value)
        # 2024.08.03 @hslee No KD Loss
        # metric_logger.update(KD_MSE=loss_nb_kd)
        metric_logger.update(loss_w_neck=loss_value_w_neck, **loss_dict_w_neck_reduced)
        # metric_logger.update(loss_wo_neck=loss_value_wo_neck, **loss_dict_wo_neck_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, wNeck):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples, wNeck=wNeck)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            
            wNeck = True
            outputs = model(samples, targets, wNeck=wNeck)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}