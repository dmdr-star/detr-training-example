import os
import sys
import logging
import random
import shutil
import warnings
import functools

from tqdm import tqdm

import numpy as np
import torch
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb


tqdm = functools.partial(
    tqdm,
    file=sys.stdout,
    bar_format="{desc}: {percentage:3.0f}%|{bar:20}{r_bar}",
    leave=True,
)


# NOTE: required for correct export to CoreML using latest PyTorch version
@register_torch_op
def type_as(context, node):
    inputs = _get_inputs(context, node)
    context.add(mb.cast(x=inputs[0], dtype="fp32"), node.name)


@register_torch_op
def silu(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    y = mb.sigmoid(x=x)
    res = mb.mul(x=x, y=y, name=node.name)
    context.add(res)


@register_torch_op
def silu_(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    y = mb.sigmoid(x=x)
    res = mb.mul(x=x, y=y, name=node.name)
    context.add(res)


INITIALIZED_LOGGERS = {}


def get_lr(optimizer):
    """Get learning rate from optimizer.

    Args:
        optimizer (torch.optim.Optimizer): model optimizer

    Returns:
        learning rate (float) if optimizer is instance of torch.optim.Optimizer,
            otherwise `None`
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        return None

    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Create logger for experiments.

    Args:
        name (str): logger name.
            If function called multiple times same name or
            name which starts with same prefix then will be
            returned initialized logger from the first call.
        log_file (str): file to use for storing logs.
            Default is `None`.
        log_level (int): logging level.
            Default is `logging.INFO`.

    Returns:
        logging.Logger object.
    """
    logger = logging.getLogger(name)
    if name in INITIALIZED_LOGGERS:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in INITIALIZED_LOGGERS:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "w")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    INITIALIZED_LOGGERS[name] = True
    return logger


def t2d(tensor, device):
    """Move tensors to a specified device.

    Args:
        tensor (torch.Tensor or Dict[str, torch.Tensor] or list/tuple of torch.Tensor):
            data to move to a device.
        device (str or torch.device): device where should be moved device

    Returns:
        torch.Tensor or Dict[str, torch.Tensor] or List[torch.Tensor] based on `tensor` type.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        # recursive move to device
        return [t2d(_tensor, device) for _tensor in tensor]
    elif isinstance(tensor, dict):
        res = {}
        for _key, _tensor in tensor.items():
            res[_key] = t2d(_tensor, device)
        return res


def seed_all(seed=42, deterministic=True, benchmark=True) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed.
            Default is `42`.
        deterministic (bool): flag to use cuda deterministic
            algoritms for computations.
            Default is `True`.
        benchmark (bool): flag to use benchmark option
            to select the best algorithm for computatins.
            Should be used `True` with fixed size
            data (images or similar) for other types of
            data is better to use `False` option.
            Default is `True`.
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = deterministic
    # small speedup
    torch.backends.cudnn.benchmark = benchmark


def make_checkpoint(stage, epoch, model, optimizer=None, scheduler=None, metrics=None, **kwargs) -> dict:
    """Generate checkpoint dict.

    Args:
        stage (str): stage name
        epoch (int): epoch index
        model (torch.nn.Module or torch.nn.DataParallel): model
        optimizer (torch.optim.Optimizer): optimizer.
            Default is ``None``.
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler.
            Default is ``None``.
        metrics (dict, optional): metrics to store in checkpoint.
            Default is ``None``.
        **kwargs: other keys with values to store in checkpoint.

    Returns:
        Dict[str, Any] with checkpoint states
    """
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return make_checkpoint(stage, epoch, model.module, optimizer, scheduler, metrics)

    if not isinstance(model, torch.nn.Module):
        raise ValueError("Expected that model will be an instance of nn.Module but got {}!".format(type(model)))

    checkpoint = {"stage": stage, "epoch": epoch}
    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics

    for key, value in kwargs.items():
        if key in checkpoint:
            warnings.warn(f"Found duplicated keyword ('{key}'), it will be ignored!")
            continue
        checkpoint[key] = value

    return checkpoint


def save_checkpoint(
    checkpoint,
    logdir,
    name,
    is_best=False,
    is_last=False,
    verbose=False,
    save_fn=torch.save,
) -> None:
    """Save checkpoint to a file.

    Args:
        checkpoint (dict): data to store in checkpoint
        logdir (str or Path): directory where should be stored checkpoint
        name (str): file name to use for storing checkpoint
        is_best (bool, optional): indicator to save checkpoint as best checkpoint.
            Defaults to False.
        is_last (bool, optional): indicator to save checkpoint as last checkpoint.
            Defaults to False.
        verbose (bool, optional): default is `False`.
        save_fn (function (callable), optional): default is `torch.save`
    """
    os.makedirs(logdir, exist_ok=True)
    _name = name if name.endswith(".pth") else f"{name}.pth"
    filename = os.path.join(str(logdir), _name)
    save_fn(checkpoint, filename)
    if verbose:
        print(f"=> Saved checkpoint '{filename}'")
    if is_best:
        best_filename = os.path.join(str(logdir), "best.pth")
        shutil.copyfile(filename, best_filename)
    if is_last:
        last_filename = os.path.join(str(logdir), "last.pth")
        shutil.copyfile(filename, last_filename)


def load_checkpoint(
    checkpoint_file,
    model,
    optimizer=None,
    scheduler=None,
    map_location=None,
    verbose=True,
):
    """Shortcut for loading checkpoint state.

    Args:
        checkpoint_file (str or Path): path to checkpoint.
        model (torch.nn.Module): model to initialize with checkpoint weights
        optimizer (torch.optim.Optimizer): optimizer to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is `None`.
        scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler to initialize with checkpoint weights.
            If `None` then will be ignored.
            Default is `None`.
        map_location (torch.device or str or dict[str, int]):
            location to use for loading checkpoint content.
            More about possible locations - `https://pytorch.org/docs/master/generated/torch.load.html`
            Default is `None`.
        verbose (bool): verbosity mode, if `True` then will print a loaded items.
            Default is `True`.
    """  # noqa: D417
    checkpoint = torch.load(str(checkpoint_file), map_location=map_location)
    loaded_items = []

    if "model_state_dict" in checkpoint and model is not None:
        state_dict = checkpoint["model_state_dict"]
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        loaded_items.append("model")

    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loaded_items.append("optimizer")

    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        loaded_items.append("scheduler")

    if loaded_items and verbose:
        print("<= Loaded {} from '{}'".format(", ".join(loaded_items), checkpoint_file))

        if "stage" in checkpoint:
            print("Stage: {}".format(checkpoint["stage"]))

        if "epoch" in checkpoint:
            print("Epoch: {}".format(checkpoint["epoch"]))

        if "metrics" in checkpoint:
            print("Metrics:")
            print(checkpoint["metrics"])


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type): module to use as replacement
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recursively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
