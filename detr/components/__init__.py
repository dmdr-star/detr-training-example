import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from detr.components.registry import Registry

OPTIMIZER_REGISTRY = Registry()
OPTIMIZER_REGISTRY.add(optim.Adam)
OPTIMIZER_REGISTRY.add(optim.AdamW)
OPTIMIZER_REGISTRY.add(optim.RMSprop)
OPTIMIZER_REGISTRY.add(optim.SGD)


SCHEDULER_REGISTRY = Registry()
SCHEDULER_REGISTRY.add(lr_scheduler.StepLR)
SCHEDULER_REGISTRY.add(lr_scheduler.CosineAnnealingLR)
SCHEDULER_REGISTRY.add(lr_scheduler.CosineAnnealingWarmRestarts)
SCHEDULER_REGISTRY.add(lr_scheduler.ReduceLROnPlateau)
SCHEDULER_REGISTRY.add(lr_scheduler.CyclicLR)


__all__ = ["OPTIMIZER_REGISTRY", "SCHEDULER_REGISTRY"]