# python libraries
import argparse
import ast
import os
import time

# torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, SequentialSampler

# our code
from libs.core import cfg
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_multimodal_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, save_checkpoint,
                        make_optimizer, make_scheduler, fix_random_seed, ModelEma, RunRecord)


def setup_distributed():
    """torchrun sets RANK / WORLD_SIZE / LOCAL_RANK and starts one process per GPU;
    a plain `python train.py` is one process on one GPU (or the CPU)."""
    if 'RANK' not in os.environ:
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
        device = torch.device('cuda', local_rank)
    else:
        dist.init_process_group('gloo')
        device = torch.device('cpu')
    return dist.get_rank(), dist.get_world_size(), device


def apply_overrides(settings):
    """--set model.article_head=True opt.epochs=8 ... : change cfg entries without editing config.py.
    The key is a dotted path into cfg; the value is a Python literal (True, 0.5, [0.9, 1.0], 'text')."""
    for setting in settings:
        key, value = setting.split('=', 1)
        path = key.split('.')
        node = cfg
        for name in path[:-1]:
            node = node[name]
        if path[-1] not in node:
            raise KeyError(f'--set {key}: no such entry in the config')
        try:
            node[path[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            node[path[-1]] = value   # a plain word, kept as a string
        print(f'config: {key} = {node[path[-1]]!r}', flush=True)


def metrics_line(m):
    """One line with the validation metrics; the ones that are -1 (head not used) are left out."""
    parts = [f"R@0.5 {m['R@0.5']:.2f}", f"mIoU {m['mIoU']:.2f}", f"Acc {m['Acc']:.2f}", f"mAP {m['mAP']:.2f}"]
    for name in ['ArtAcc', 'OmniCos', 'OmniCosC', 'OmniR@1']:
        if m.get(name, -1) >= 0:
            parts.append(f"{name} {m[name]:.2f}")
    return '  '.join(parts)


def main(args):
    """main function: train, then evaluate after every epoch"""

    """1. parameters / folders"""
    # ======================================= LOAD CONFIG =======================================================
    rank, world_size, device = setup_distributed()
    is_main = rank == 0   # only process 0 evaluates, saves checkpoints and prints
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    apply_overrides(args.set)
    if args.output:
        cfg['output_folder'] = os.path.join('./ckpt', args.output)
    ckpt_folder = cfg['output_folder']
    

    """2. datasets / dataloaders"""
    # ======================================= LOADER =======================================================
    data_cfg = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
                    regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
    
    train_dataset = make_dataset('htstep', True, ['train'], **data_cfg)
    val_dataset = make_dataset('htstep', False, ['val_seen'], **data_cfg)
    
    # every process trains on its own share of the training videos, reshuffled every epoch
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                       seed=cfg['init_rand_seed'], drop_last=True)
    train_loader = make_data_loader(train_dataset, True, rng_generator, cfg['batch_size'], cfg['num_workers'],
                                    train_sampler)
    val_loader = None
    if is_main:
        val_loader = make_data_loader(val_dataset, False, None, cfg['batch_size'], cfg['num_workers'],
                                      SequentialSampler(val_dataset))

    # print('Data: %d train / %d val videos, %d unique train captions'
    #       % (len(train_dataset), len(val_dataset), len(train_dataset.pool_text)))

    """3. model, optimizer and scheduler"""
    # ======================================= MODEL CALLED HERE =======================================================
    model = make_multimodal_meta_arch('EventCaptionTransformer', **cfg['model'])
    model.to(device)
    # the classes: text vectors of every step of the taxonomy (no annotation needed)
    model.set_step_pool(
        torch.from_numpy(train_dataset.pool_emb).to(device), # (N, 512)
        [train_dataset.article_steps[a] for a in train_dataset.articles] # pool rows of every article's steps
    )
    # mean of the train teacher vectors, removed by the OmniCosC metric
    if model.use_omni:
        model.omni_teacher_mean = torch.from_numpy(train_dataset.omni_teacher_mean()).to(device)
    # print('Parameters: %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    if args.eval:   # score a checkpoint, no training
        if is_main:
            model.load_state_dict(torch.load(args.eval, map_location=device)['state_dict'])
            m = valid_one_epoch(val_loader, model, train_dataset.pool_text, device)
            print(f"{args.eval}: {metrics_line(m)}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    # DDP copies the weights of process 0 to every process, and averages the gradients over the processes
    net = model
    if world_size > 1:
        net = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    optimizer = make_optimizer(model, cfg['opt'])
    scheduler = make_scheduler(optimizer, cfg['opt'], len(train_loader))
    model_ema = ModelEma(model, cfg['train_cfg']['ema_decay'])
    model_ema.module.set_step_pool(model.pool_raw, model.article_steps)
    model_ema.module.omni_teacher_mean = model.omni_teacher_mean
    record = RunRecord(cfg, world_size) if is_main else None

    """4. train / evaluate loop"""
    max_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    # checkpoint metric: the official article grounding mAP. R@0.5 (top-k) penalises real steps HT-Step leaves
    # unannotated (GT covers ~20% of a video), so it is only logged.
    # Early stop: a metric counts as improved only if it beats its best by at least --min_delta
    # (a gain of 0.01 means nothing); the run stops when none of the watched metrics improved for --patience epochs.
    watched = ['mAP', 'ArtAcc', 'OmniCos']
    best = {name: (-1.0, 0) for name in watched}   # metric -> (best value, its epoch)
    ckpt_path = os.path.join(ckpt_folder, 'model_best.pth.tar')
    last_improvement = 0
    for epoch in range(max_epochs):
        start = time.time()
        train_sampler.set_epoch(epoch)
        losses = train_one_epoch(train_loader, net, optimizer, scheduler, model_ema,
                                 cfg['train_cfg']['clip_grad_l2norm'], device)
        stop = False
        if is_main:
            m = valid_one_epoch(val_loader, model_ema.module, train_dataset.pool_text, device)
            marks = []
            for name in watched:
                if m[name] < 0:
                    continue   # metric not available (no article head / no omni head)
                if m[name] >= best[name][0] + args.min_delta:
                    best[name] = (m[name], epoch)
                    last_improvement = epoch
                    marks.append(name)
            if 'mAP' in marks:
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model_ema.module.state_dict(), 'metrics': m},
                                ckpt_folder, 'model_best.pth.tar')
            if 'ArtAcc' in marks:
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model_ema.module.state_dict(), 'metrics': m},
                                ckpt_folder, 'model_best_article.pth.tar')
            mark = f"  <- better {', '.join(marks)}" if marks else ''
            print(f"epoch {epoch + 1:2d}/{max_epochs}  {time.time() - start:4.0f}s  loss {losses['final_loss']:.3f}"
                  f" | {metrics_line(m)}{mark}", flush=True)
            record.add_epoch(epoch + 1, time.time() - start, losses, m,
                             {k: {'value': v, 'epoch': e + 1} for k, (v, e) in best.items()})
            stop = args.patience > 0 and epoch - last_improvement >= args.patience
        if world_size > 1:   # every process follows the decision of process 0
            flag = [stop]
            dist.broadcast_object_list(flag, src=0)
            stop = flag[0]
        if stop:
            if is_main:
                print(f'Early stop: no metric improved by {args.min_delta} for {args.patience} epochs', flush=True)
            break

    if is_main:
        print(f"Best mAP = {best['mAP'][0]:.2f} -> {ckpt_path}")
        record.finish(stopped_early=stop, ckpt=ckpt_path)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train on HT-Step; parameters are in libs/core/config.py. '
                                            'Several GPUs: torchrun --nproc_per_node=N train.py')
    p.add_argument('--eval', default='', help='only score this checkpoint, no training')
    p.add_argument('--output', default='', help='run name: checkpoints go to ckpt/<output> (default: config output_folder)')
    p.add_argument('--set', nargs='+', action='extend', default=[], help='config overrides, e.g. model.article_head=True opt.epochs=8')
    p.add_argument('--patience', type=int, default=2,
                   help='stop when no watched metric (mAP, ArtAcc, OmniCos) has improved for this many epochs (0 = never)')
    p.add_argument('--min_delta', type=float, default=0.3,
                   help='a metric counts as improved only if it beats its best by at least this much')
    main(p.parse_args())
