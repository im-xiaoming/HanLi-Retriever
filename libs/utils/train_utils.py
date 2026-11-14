import os
import copy
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .lr_schedulers import LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    """Seed python, numpy and torch; with include_cuda also make cuDNN / cuBLAS deterministic.
    Returns the torch generator for the DataLoader."""
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if include_cuda:
        # no cudnn benchmark, so a rerun gives the same results
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    return rng_generator


def save_checkpoint(state, file_folder, file_name):
    """write a checkpoint to a file"""
    os.makedirs(file_folder, exist_ok=True)
    torch.save(state, os.path.join(file_folder, file_name))


def make_optimizer(model, optimizer_config):
    """AdamW with three parameter groups (as ActionFormer, plus one for the caption projection):
      decay    : weights of the Linear / Conv layers, with weight decay
      no_decay : biases, LayerNorm, scales and other parameters, no weight decay
      clip     : clip_proj (caption vectors into the event space), its own learning rate clip_proj_lr
    """
    decay, no_decay, clip = [], [], []
    for mod_name, mod in model.named_modules():
        for par_name, p in mod.named_parameters(recurse=False):
            if 'clip_proj' in mod_name:
                clip.append(p)
            elif par_name.endswith('weight') and isinstance(mod, (torch.nn.Linear, torch.nn.Conv1d)):
                decay.append(p)
            else:
                no_decay.append(p)
    groups = [
        {'params': decay, 'weight_decay': optimizer_config['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': clip, 'weight_decay': 0.0, 'lr': optimizer_config['clip_proj_lr']},
    ]
    return torch.optim.AdamW(groups, lr=optimizer_config['learning_rate'])


def make_scheduler(optimizer, optimizer_config, num_iters_per_epoch):
    """Linear warm-up over warmup_epochs, then a cosine decay; steps after every iteration."""
    warmup_steps = optimizer_config['warmup_epochs'] * num_iters_per_epoch
    max_steps = (optimizer_config['epochs'] + optimizer_config['warmup_epochs']) * num_iters_per_epoch
    return LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)


class ModelEma(torch.nn.Module):
    """Exponential moving average (EMA) of the weights, as ActionFormer: evaluation and checkpoints
    both use these averaged weights."""

    def __init__(self, model, decay=0.999):
        super().__init__()
        # a copy of the model accumulates the moving average
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        # the decay ramps up: with a fixed 0.999 the weights are still almost the initial ones after the
        # first epoch, and the first evaluations would score a model that has learnt nothing
        model = getattr(model, 'module', model)   # the model inside DistributedDataParallel
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v.detach(), alpha=1 - d)
            else:
                ema_v.copy_(model_v)


def train_one_epoch(
    train_loader,      # DataLoader, every batch is a list of dicts (see HTStepDataset.__getitem__)
    model,             # the model, or the DistributedDataParallel around it
    optimizer,
    scheduler,
    model_ema,         # ModelEma, updated after every step
    clip_grad_l2norm,  # maximum L2 norm of the gradients
    device,
):
    """Train one epoch, returns the mean of every loss (over this process's batches)."""
    model.train()
    losses = {}
    for video_list in train_loader:
        for x in video_list:
            x['feats'] = {k: v.to(device) for k, v in x['feats'].items()}
        losses_dict = model(video_list)
        optimizer.zero_grad(set_to_none=True)
        losses_dict['final_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        optimizer.step()
        scheduler.step()   # the learning rate steps after every iteration, not every epoch
        model_ema.update(model)
        for k, v in losses_dict.items():
            losses[k] = losses.get(k, 0.0) + v.item() / len(train_loader)
    return losses
