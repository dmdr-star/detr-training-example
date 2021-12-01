import warnings

import numpy as np
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def _box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = _box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def change_box_order(boxes, order):
    """Change box order between
    (xmin, ymin, xmax, ymax) <-> (xcenter, ycenter, width, height).

    Args:
        boxes: (torch.Tensor or np.ndarray) bounding boxes, sized [N,4].
        order: (str) either "xyxy2xywh" or "xywh2xyxy".

    Returns:
        (torch.Tensor) converted bounding boxes, sized [N,4].
    """
    if order not in {"xyxy2xywh", "xywh2xyxy"}:
        raise ValueError("`order` should be one of 'xyxy2xywh'/'xywh2xyxy'!")

    concat_fn = torch.cat if isinstance(boxes, torch.Tensor) else np.concatenate

    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return concat_fn([(a + b) / 2, b - a], 1)
    return concat_fn([a - b / 2, a + b / 2], 1)


def _numpy_box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (np.ndarray) bounding boxes with shape [N, 4].
        box2: (np.ndarray) bounding boxes with shape [M, 4].

    Return:
        iou (np.ndarray): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)  # [N, M]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iou = inter / (area1[:, None] + area2 - inter)
    return iou


def _torch_box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (torch.Tensor) bounding boxes with shape [N, 4].
        box2: (torch.Tensor) bounding boxes with shape [M, 4].

    Return:
        iou (torch.Tensor): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_iou(box1, box2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        box1: (torch.Tensor or np.ndarray) bounding boxes with shape [N,4].
        box2: (torch.Tensor or np.ndarray) bounding boxes with shape [M,4].

    Return:
        iou (torch.Tensor or np.ndarray): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor):
        return _torch_box_iou(box1, box2)
    else:
        return _numpy_box_iou(box1, box2)


def _torch_box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes: (torch.Tensor) bounding boxes with shape [N,4].
        scores: (torch.Tensor) confidence scores with shape [N,].
        threshold: (float) overlap threshold.

    Returns:
        keep: (torch.Tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    if bboxes.dim() == 1:
        bboxes = bboxes.reshape(-1, 4)

    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.dim() == 0:
            i = order.item()
        else:
            i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long)


def _numpy_box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes (np.ndarray): bounding boxes, sized [N,4].
        scores (np.ndarray): confidence scores, sized [N,].
        threshold (float): overlap threshold.

    Returns:
        keep: (List[int]) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

    return keep


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes (torch.Tensor): bounding boxes with shape [N,4].
        scores (torch.Tensor): confidence scores with shape [N,].
        threshold (float): overlap threshold.

    Returns:
        keep: (torch.Tensor) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    if isinstance(bboxes, torch.Tensor) and isinstance(scores, torch.Tensor):
        return _torch_box_nms(bboxes, scores, threshold)
    else:
        return _numpy_box_nms(bboxes, scores, threshold)


def nms_filter(bboxes, classes, confidences, iou_threshold=0.5):
    """Filter classes, bboxes, confidences using nms with iou_threshold.

    Args:
        bboxes (np.ndarray): array with bounding boxes, expected shape [N, 4].
        classes (np.ndarray): array with classes, expected shape [N,].
        confidences (np.ndarray)): array with class confidence, expected shape [N,].
        iou_threshold (float): IoU threshold to use for filtering.
            Default is ``0.5``.

    Returns:
        filtered bboxes (np.ndarray), classes (np.ndarray), and confidences (np.ndarray)
            where number of records will be equal to some M (M <= N).
    """
    keep_bboxes = []
    keep_classes = []
    keep_confidences = []

    for presented_cls in np.unique(classes):
        mask = classes == presented_cls
        curr_bboxes = bboxes[mask, :]
        curr_classes = classes[mask]
        curr_confs = confidences[mask]

        to_keep = box_nms(curr_bboxes, curr_confs, iou_threshold)

        keep_bboxes.append(curr_bboxes[to_keep, :])
        keep_classes.append(curr_classes[to_keep])
        keep_confidences.append(curr_confs[to_keep])

    return np.concatenate(keep_bboxes), np.concatenate(keep_classes), np.concatenate(keep_confidences)
