from .models import make_multimodal_meta_arch
from . import multimodal_backbones   # registers the backbone
from . import event_archs            # registers the model

__all__ = ['make_multimodal_meta_arch']
