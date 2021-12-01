import json

import numpy as np
import torch

from detr import metrics
from detr import modeling
from detr import utils
from detr import components
from detr import datasets


LOGGER = None
IMG_SIZE = None
IMG_MEAN = None
IMG_STD = None


def train_fn(loader, model, criterion, device, optimizer, scheduler=None, verbose=True):
    """Train model on a specified loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader.
        model (torch.nn.Module): model to use for training.
        device (str/int/torch.device): device to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): batch scheduler
            (will be triggered after each batch).
            Default is `None`.
        verbose (bool): option to print information about training progress.
            Default is `True`.

    Returns:
        Metrics (dict where key (str) is a metric name and value (float))
        collected during the training on specified loader
    """
    model.train()
    criterion.train()
    num_batches = len(loader)
    metrics = {
        "loss": 0.0,
        "loss_ce": 0.0,
        "class_error": 0.0,
        "loss_bbox": 0.0,
        "loss_giou": 0.0,
        "cardinality_error": 0.0,
    }
    progress_str = (
        "loss - {loss:.4f}, loss_ce - {loss_ce:.4f}, "
        "class_error - {class_error:.4f}, loss_bbox - {loss_bbox:.4f}, "
        "loss_giou - {loss_giou:.4f}, cardinality_error - {cardinality_error:.4f}"
    )
    with utils.misc.tqdm(desc="train", total=num_batches, disable=not verbose) as progress:
        for images, targets in loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)

            _loss_dict_values = {"loss": loss.item()}
            for k in ("loss_ce", "class_error", "loss_bbox", "loss_giou", "cardinality_error"):
                _loss_dict_values[k] = loss_dict[k].item()

            for k in ("loss", "loss_ce", "class_error", "loss_bbox", "loss_giou", "cardinality_error"):
                metrics[k] += _loss_dict_values[k]

            optimizer.zero_grad()
            loss.backward()

            progress.set_postfix_str(progress_str.format(**_loss_dict_values))

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            progress.update(1)
    if num_batches != 0:
        metrics = {k: (v / num_batches) for k, v in metrics.items()}
    metrics["lr"] = utils.misc.get_lr(optimizer)
    return metrics


@torch.no_grad()
def valid_fn(loader, model, device, num_classes, class_labels, verbose=True):
    """Validate model on a specified loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader.
        model (torch.nn.Module): model to use for training.
        device (str/int/torch.device): device to use for training.
        num_classes (int): number of classes in dataset.
        class_labels (List[str]): class labels.
        verbose (bool): option to print information about training progress.
            Default is `True`.

    Returns:
        Metrics (dict where key (str) is a metric name and value (float))
        collected during the validation on specified loader.
    """
    model.eval()
    num_batches = len(loader)
    mean_ap = metrics.MeanAveragePrecision(num_classes)
    conf_matrix = metrics.ConfusionMatrix(num_classes)
    with utils.misc.tqdm(desc="validation", total=num_batches, disable=not verbose) as progress:
        for images, targets in loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_size = len(images.tensors)
            _, width, height = images.tensors[0].size()

            outputs = model(images)

            probs = outputs["pred_logits"].softmax(-1).detach().cpu().numpy()
            pred_cls = np.argmax(probs, -1)
            pred_confidence = np.max(probs, -1)

            pred_boxes = outputs["pred_boxes"].detach().cpu().numpy()
            # pred_boxes = (change_box_order(pred_boxes, "xywh2xyxy") * (width, height, width, height)).astype(np.int32)

            for i in range(batch_size):
                # build predictions
                sample_bboxes = utils.box_ops.change_box_order(pred_boxes[i], "xywh2xyxy")
                sample_bboxes = (sample_bboxes * (width, height, width, height)).astype(np.int32)
                sample_bboxes, sample_classes, sample_confs = utils.box_ops.nms_filter(
                    sample_bboxes, pred_cls[i], pred_confidence[i]
                )
                pred_sample = np.concatenate([sample_bboxes, sample_classes[:, None], sample_confs[:, None]], -1)
                pred_sample = pred_sample.astype(np.float32)

                # build ground truth
                sample_gt_bboxes = utils.box_ops.change_box_order(
                    targets[i]["boxes"].detach().cpu().numpy(), "xywh2xyxy"
                )
                sample_gt_bboxes = (sample_gt_bboxes * (width, height, width, height)).astype(np.int32)
                sample_gt_classes = targets[i]["labels"].detach().cpu().numpy()
                gt_sample = np.zeros((sample_gt_classes.shape[0], 7), dtype=np.float32)
                gt_sample[:, :4] = sample_gt_bboxes
                gt_sample[:, 4] = sample_gt_classes

                # update metrics statistics
                mean_ap.add(pred_sample, gt_sample)
                conf_matrix.add(pred_sample, gt_sample[:, :5])

            progress.update(1)
    # fmt: off
    metrics_dict = {
        "mAP": mean_ap.value()["mAP"],
        "confusion_matrix": {
            "matrix": conf_matrix.value().tolist(),
            "labels": class_labels,
        },
    }
    # fmt: on
    return metrics_dict


def experiment(device, args):
    global LOGGER, IMG_SIZE, IMG_MEAN, IMG_STD

    args = dict() if args is None else args
    #
    verbose = args["progress"]
    #
    architecture = args["model"]["arch"]
    IMG_SIZE = args["model"]["image_sizes"]
    IMG_MEAN = args["model"]["image_mean"]
    IMG_STD = args["model"]["image_std"]
    num_classes = args["model"]["num_classes"]
    #
    num_epochs = args["experiment"]["num_epochs"]
    validation_period = args["experiment"]["validation_period"]
    augmentations_intensity = args["experiment"].get("augmentations", "basic")
    metrics_file = args["experiment"]["metrics_log_file"]
    mapping_file = args["experiment"]["class_mapping"]
    checkpoint_file = args["experiment"]["checkpoint_file"]
    onnx_file = args["experiment"].get("onnx_file")
    coreml_file = args["experiment"].get("coreml_file")
    tflite_file = args["experiment"].get("tflite_file")

    LOGGER = utils.misc.get_logger(architecture)
    LOGGER.info(f"Experiment arguments:\n{json.dumps(args, indent=4)}")
    METRICS_LOGGER = utils.loggers.JSONLogger(metrics_file)

    #######################################################################
    # datasets
    #######################################################################

    train_loader, valid_loader = datasets.get_loaders(
        train_annotations=args["train"]["annotations"],
        train_images_dir=args["train"]["images_dir"],
        train_batch_size=args["train"]["batch_size"],
        train_num_workers=args["train"]["num_workers"],
        #
        valid_annotations=args["validation"]["annotations"],
        valid_images_dir=args["validation"]["images_dir"],
        valid_batch_size=args["validation"]["batch_size"],
        valid_num_workers=args["validation"]["num_workers"],
        #
        image_size=IMG_SIZE,
        image_mean=IMG_MEAN,
        image_std=IMG_STD,
        augmentations_intensity=augmentations_intensity,
    )

    LOGGER.info("Train dataset information:\n" + train_loader.dataset.info())
    LOGGER.info("Validation dataset information:\n" + valid_loader.dataset.info())
    class_labels = train_loader.dataset.class_labels + ["<NO MATCHES>"]
    n_objects = max(
        train_loader.dataset.dataset_stats["Max number of objects on one image"],
        valid_loader.dataset.dataset_stats["Max number of objects on one image"],
    )

    #######################################################################
    # experiment parts
    #######################################################################

    utils.misc.seed_all(42)
    model = modeling.get_model(architecture, IMG_SIZE[0], num_classes, n_objects=n_objects)
    model = model.to(device)
    criterion = modeling.build_criterion(num_classes)
    criterion = criterion.to(device)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    ]
    optimizer = components.OPTIMIZER_REGISTRY[args["optimizer"]["name"]](param_dicts, **args["optimizer"]["args"])
    epoch_scheduler = components.SCHEDULER_REGISTRY[args["scheduler"]["name"]](optimizer, **args["scheduler"]["args"])
    batch_scheduler = None

    #######################################################################
    # train loop
    #######################################################################

    LOGGER.info("Started training")
    for epoch_idx in range(1, num_epochs + 1):
        LOGGER.info(f"Epoch: {epoch_idx}/{num_epochs}")
        train_metrics = train_fn(train_loader, model, criterion, device, optimizer, batch_scheduler, verbose=verbose)
        utils.loggers.log_metrics(LOGGER, train_metrics, epoch_idx, "\nTrain:", loader="train")
        utils.loggers.log_metrics(METRICS_LOGGER, train_metrics, epoch_idx, loader="train")

        # do validation, if required
        if epoch_idx % validation_period == 0:
            valid_metrics = valid_fn(valid_loader, model, device, num_classes, class_labels, verbose=verbose)
            utils.loggers.log_metrics(LOGGER, valid_metrics, epoch_idx, "\nValidation:", loader="validation")
            utils.loggers.log_metrics(METRICS_LOGGER, valid_metrics, epoch_idx, loader="validation")

        # change lr after epoch
        epoch_scheduler.step()

    #######################################################################
    # experiment artifacts
    #######################################################################

    model_info = {
        "model": {
            "name": architecture,
            "input_shapes": {
                "batch": 1,
                "channels": 3,
                "height": IMG_SIZE[0],
                "width": IMG_SIZE[1],
            },
            "image": {
                "mean": IMG_MEAN,
                "std": IMG_STD,
            },
            "classes": train_loader.dataset.get_class_mapping(),
        }
    }
    with open(mapping_file, "w") as f:
        json.dump(model_info, f, indent=4)
    LOGGER.info(f"Saved labels mapping to '{mapping_file}'")

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_file)
    LOGGER.info(f"Saved PyTorch model to '{checkpoint_file}'")

    model = modeling.get_model(architecture, IMG_SIZE[0], num_classes, n_objects=n_objects)
    model.load_state_dict(torch.load(checkpoint_file, map_location="cpu")["model_state_dict"])

    if onnx_file:
        model.save_onnx(model, torch.randn(1, 3, *IMG_SIZE), onnx_file)
        LOGGER.info(f"Exported ONNX model to '{onnx_file}'")

    if coreml_file:
        model.save_coreml(model, torch.randn(1, 3, *IMG_SIZE), coreml_file)
        LOGGER.info(f"Exported CoreML model to '{coreml_file}'")

    if tflite_file:
        model.save_tflite(model, torch.randn(1, 3, *IMG_SIZE), tflite_file)
        LOGGER.info(f"Exported TFLite model to '{tflite_file}'")
