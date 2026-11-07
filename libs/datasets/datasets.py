import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed
from typing import Literal

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

# location generators (points, segments, ...)
generators = {}
def register_generator(name):
    def decorator(cls):
        generators[name] = cls
        return cls
    return decorator


# kwargs = dict(cfg['dataset'], scale_factor=cfg['model']['scale_factor'],
#                     regression_range=cfg['model']['regression_range'], backbone_arch=cfg['model']['backbone_arch'])
def make_dataset(name, is_training, split, **kwargs):
   """
       Build a registered dataset by name
   """
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers, sampler):
    """
        Build the DataLoader. Training drops the last incomplete batch and reseeds every worker
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=trivial_batch_collator, # do nothing
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        drop_last=is_training,
        generator=generator,
        persistent_workers=num_workers > 0
    )
    return loader

def make_generator(name, **kwargs):
    generator = generators[name](**kwargs)
    return generator
