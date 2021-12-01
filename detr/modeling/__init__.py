from functools import partial

from detr.modeling.criterion import build_criterion  # noqa: F401
from detr.modeling.model import build_detr  # noqa: F401


_models = {
    "detr-resnet18": partial(build_detr, backbone="resnet18", num_encoder_layers=2, num_decoder_layers=2),
    "detr-resnet34": partial(build_detr, backbone="resnet34", num_encoder_layers=6, num_decoder_layers=6),
    "detr-resnet50": partial(build_detr, backbone="resnet50", num_encoder_layers=6, num_decoder_layers=6),
    "detr-resnet101": partial(build_detr, backbone="resnet101", num_encoder_layers=12, num_decoder_layers=12),
    "detr-resnet152": partial(build_detr, backbone="resnet152", num_encoder_layers=12, num_decoder_layers=12),
}


def list_models():
    """Information about supported models.

    Returns:
        List[str] with supported detectors.
    """
    return sorted(_models.keys())


def get_model(arch, img_size, num_classes, pretrained=False, **kwargs):
    """Create model from arguments.

    Args:
        arch (str): model architecture.
        img_size (int): image size.
        num_classes (int): number of classes/objects to predict.
        pretrained (bool): option to use pretrained weights.
            Default is `False`.

    Returns:
        model (torch.nn.Module)
    """
    model_fn = _models.get(arch)
    if model_fn is None:
        raise RuntimeError(f"Unknown architecture - '{arch}'!")
    return model_fn(img_size=img_size, num_classes=num_classes, pretrained=pretrained, **kwargs)
