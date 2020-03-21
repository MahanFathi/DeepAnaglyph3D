from model import archs
from model import optimizers

def build_model(cfg):
    arch_factory = getattr(archs, cfg.MODEL.ARCH)
    optimizer_factory = getattr(optimizers, cfg.MODEL.ARCH)
    model = arch_factory(cfg)
    optimizer = optimizer_factory(cfg, model)
    return model, optimizer
