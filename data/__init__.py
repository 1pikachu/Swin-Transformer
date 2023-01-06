from .build import build_loader as _build_loader
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, simmim=False, is_pretrain=False, device="cpu"):
    if not simmim:
        return _build_loader(config, device)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
