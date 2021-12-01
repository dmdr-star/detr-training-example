import warnings

import numpy as np
import pandas as pd

# Source: https://github.com/bes-dev/mean_average_precision


def row_to_vars(row):
    """Convert row of pd.DataFrame to variables.

    Arguments:
        row (pd.DataFrame): row

    Returns:
        img_id (int): image index.
        conf (flaot): confidence of predicted box.
        iou (np.array): iou between predicted box and gt boxes.
        difficult (np.array): difficult of gt boxes.
        crowd (np.array): crowd of gt boxes.
        order (np.array): sorted order of iou's.
    """
    img_id = row["img_id"]
    conf = row["confidence"]
    iou = np.array(row["iou"])
    difficult = np.array(row["difficult"])
    crowd = np.array(row["crowd"])
    order = np.argsort(iou)[::-1]
    return img_id, conf, iou, difficult, crowd, order


def compute_match_table(preds, gt, img_id):
    """Compute match table.

    Args:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.
        img_id (int): image id

    Returns:
        match_table (pd.DataFrame)

    Input format:
        preds: [xmin, ymin, xmax, ymax, class_id, confidence]
        gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]

    Output format:
        match_table: [img_id, confidence, iou, difficult, crowd]
    """

    def _tile(arr, nreps, axis=0):
        return np.repeat(arr, nreps, axis=axis).reshape(nreps, -1).tolist()

    def _empty_array_2d(size):
        return [[] for i in range(size)]

    match_table = {}
    match_table["img_id"] = [img_id for i in range(preds.shape[0])]
    match_table["confidence"] = preds[:, 5].tolist()
    if gt.shape[0] > 0:
        match_table["iou"] = compute_iou(preds, gt).tolist()
        match_table["difficult"] = _tile(gt[:, 5], preds.shape[0], axis=0)
        match_table["crowd"] = _tile(gt[:, 6], preds.shape[0], axis=0)
    else:
        match_table["iou"] = _empty_array_2d(preds.shape[0])
        match_table["difficult"] = _empty_array_2d(preds.shape[0])
        match_table["crowd"] = _empty_array_2d(preds.shape[0])
    return pd.DataFrame(match_table, columns=list(match_table.keys()))


def check_box(iou, difficult, crowd, order, matched_ind, iou_threshold, mpolicy="greedy"):
    """Check box for tp/fp/ignore.

    Args:
        iou (np.array): iou between predicted box and gt boxes.
        difficult (np.array): difficult of gt boxes.
        order (np.array): sorted order of iou's.
        matched_ind (list): matched gt indexes.
        iou_threshold (flaot): iou threshold.
        mpolicy (str): box matching policy.
            ``"greedy"`` - greedy matching like VOC PASCAL.
            ``"soft"`` - soft matching like COCO.
    """
    assert mpolicy in ["greedy", "soft"]

    if len(order):
        result = ("fp", -1)
        n_check = 1 if mpolicy == "greedy" else len(order)
        for i in range(n_check):
            idx = order[i]
            if iou[idx] > iou_threshold:
                if not difficult[idx]:
                    if idx not in matched_ind:
                        result = ("tp", idx)
                        break
                    elif crowd[idx]:
                        result = ("ignore", -1)
                        break
                    else:
                        continue
                else:
                    result = ("ignore", -1)
                    break
            else:
                result = ("fp", -1)
                break
    else:
        result = ("fp", -1)
    return result


def compute_iou(pred, gt):
    """Calculates IoU (Jaccard index) of two sets of bboxes:
            IOU = pred ∩ gt / (area(pred) + area(gt) - pred ∩ gt)

    Args:
        Coordinates of bboxes are supposed to be in the following form: [x1, y1, x2, y2]
        pred (np.array): predicted bboxes
        gt (np.array): ground truth bboxes

    Returns:
        iou (np.array): intersection over union
    """

    def get_box_area(box):
        return (box[:, 2] - box[:, 0] + 1.0) * (box[:, 3] - box[:, 1] + 1.0)

    _gt = np.tile(gt, (pred.shape[0], 1))
    _pred = np.repeat(pred, gt.shape[0], axis=0)

    ixmin = np.maximum(_gt[:, 0], _pred[:, 0])
    iymin = np.maximum(_gt[:, 1], _pred[:, 1])
    ixmax = np.minimum(_gt[:, 2], _pred[:, 2])
    iymax = np.minimum(_gt[:, 3], _pred[:, 3])

    width = np.maximum(ixmax - ixmin + 1.0, 0)
    height = np.maximum(iymax - iymin + 1.0, 0)

    intersection_area = width * height
    union_area = get_box_area(_gt) + get_box_area(_pred) - intersection_area
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iou = (intersection_area / union_area).reshape(pred.shape[0], gt.shape[0])
    return iou


def compute_precision_recall(tp, fp, n_positives):
    """Compute Preision/Recall.

    Args:
        tp (np.array): true positives array.
        fp (np.array): false positives.
        n_positives (int): num positives.

    Returns:
        precision (np.array)
        recall (np.array)
    """
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(float(n_positives), 1)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return precision, recall


def compute_average_precision(precision, recall):
    """Compute Avearage Precision by all points.

    Args:
        precision (np.array): precision values.
        recall (np.array): recall values.

    Returns:
        average_precision (np.array)
    """
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    ids = np.where(recall[1:] != recall[:-1])[0]
    average_precision = np.sum((recall[ids + 1] - recall[ids]) * precision[ids + 1])
    return average_precision


def compute_average_precision_with_recall_thresholds(precision, recall, recall_thresholds):
    """Compute Avearage Precision by specific points.

    Args:
        precision (np.array): precision values.
        recall (np.array): recall values.
        recall_thresholds (np.array): specific recall thresholds.

    Returns:
        average_precision (np.array)
    """
    average_precision = 0.0
    for t in recall_thresholds:
        p = np.max(precision[recall >= t]) if np.sum(recall >= t) != 0 else 0
        average_precision = average_precision + p / recall_thresholds.size
    return average_precision


class MeanAveragePrecision:
    """Mean Average Precision for bboxes.

    Example:

    .. code-block:: python

        import numpy as np
        ground_truth = np.array([
            # x_min, y_min, x_max, y_max, class_id, is_difficult, is_crowd
            [439, 157, 556, 241, 0, 0, 0],
            [437, 246, 518, 351, 0, 0, 0],
            [515, 306, 595, 375, 0, 0, 0],
            [407, 386, 531, 476, 0, 0, 0],
            [544, 419, 621, 476, 0, 0, 0],
            [609, 297, 636, 392, 0, 0, 0],
        ])
        predicted = np.array([
            # x_min, y_min, x_max, y_max, class_id, confidence
            [429, 219, 528, 247, 0, 0.460851],
            [433, 260, 506, 336, 0, 0.269833],
            [518, 314, 603, 369, 0, 0.462608],
            [592, 310, 634, 388, 0, 0.298196],
            [403, 384, 517, 461, 0, 0.382881],
            [405, 429, 519, 470, 0, 0.369369],
            [433, 272, 499, 341, 0, 0.272826],
            [413, 390, 515, 459, 0, 0.619459],
        ])
        metric_fn = MeanAveragePrecision(num_classes=1)
        for _ in range(10):
            metric_fn.add(predicted, ground_truth)
        print("mAP -", metric_fn.value(iou_thresholds=0.5)["mAP"])
        # Output:
        #   mAP - 0.5
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (float): number of classes in data
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset stored data."""
        self.image_statistics = 0
        self.class_statistics = np.zeros((0, self.num_classes), dtype=np.int32)
        cols = ["img_id", "confidence", "iou", "difficult", "crowd"]
        self.match_table = []
        for _ in range(self.num_classes):
            self.match_table.append(pd.DataFrame(columns=cols))

    def add(self, predictions, ground_truth):
        """Add sample to evaluation.

        Args:
            predictions (np.ndarray): prediction bboxes,
                Expected shapes - [N, 6] where N is a anchors.
                Each row should have structure [x_min, y_min, x_max, y_max, class_id, confidence].
            ground_truth (np.ndarray): actual bounding boxes.
                Expected shapes - [M, 7] where M is a number of bboxes on image.
                Each row should have structure [x_min, y_min, x_max, y_max, class_id, difficult, iscrowd].
        """
        assert predictions.ndim == 2 and predictions.shape[1] == 6
        assert ground_truth.ndim == 2 and ground_truth.shape[1] == 7

        class_counter = np.zeros((1, self.num_classes), dtype=np.int32)
        for c in range(self.num_classes):
            gt_c = ground_truth[ground_truth[:, 4] == c]
            class_counter[0, c] = gt_c.shape[0]  # num records
            preds_c = predictions[predictions[:, 4] == c]
            if preds_c.shape[0] > 0:
                match_table = compute_match_table(preds_c, gt_c, self.image_statistics)
                self.match_table[c] = self.match_table[c].append(match_table)

        self.image_statistics = self.image_statistics + 1
        self.class_statistics = np.concatenate((self.class_statistics, class_counter), axis=0)

    def value(self, iou_thresholds=[0.5], recall_thresholds=None, mpolicy="greedy"):
        """Compute Mean Average Precision.

        Args:
            iou_thresholds (List[float] or float): IOU thresholds.
                Default is ``[0.5]``.
            recall_thresholds (np.ndarray): specific recall thresholds
                to use for computation of average precision.
                Default is ``None``.
            mpolicy (str): box matching policy. Should be one of
                ``"greedy"`` (greedy matching like in PASCAL VOC) or
                ``"soft"`` (soft matching like in COCO).

        Returns:
            metric (dict): evaluated metrics.

        Output format:
            {
                "mAP": float.
                "<iou_threshold_0>":
                {
                    "<cls_id>":
                    {
                        "ap": float,
                        "precision": np.array,
                        "recall": np.array,
                    }
                },
                ...
                "<iou_threshold_N>":
                {
                    "<cls_id>":
                    {
                        "ap": float,
                        "precision": np.array,
                        "recall": np.array,
                    }
                }
            }
        """
        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]

        metric = {}
        aps = np.zeros((0, self.num_classes), dtype=np.float32)
        for t in iou_thresholds:
            metric[t] = {}
            aps_t = np.zeros((1, self.num_classes), dtype=np.float32)
            for class_id in range(self.num_classes):
                aps_t[0, class_id], precision, recall = self._evaluate_class(class_id, t, recall_thresholds, mpolicy)
                metric[t][class_id] = {}
                metric[t][class_id]["ap"] = aps_t[0, class_id]
                metric[t][class_id]["precision"] = precision
                metric[t][class_id]["recall"] = recall
            aps = np.concatenate((aps, aps_t), axis=0)
        metric["mAP"] = aps.mean(axis=1).mean(axis=0)
        return metric

    def _evaluate_class(self, class_id, iou_threshold, recall_thresholds, mpolicy="greedy"):
        """Evaluate class.

        Args:
            class_id (int): index of evaluated class.
            iou_threshold (float): iou threshold.
            recall_thresholds (np.ndarray): specific recall thresholds
                to use for computation of average precision.
                Default is ``None``.
            mpolicy (str): box matching policy. Should be one of
                ``"greedy"`` (greedy matching like in PASCAL VOC) or
                ``"soft"`` (soft matching like in COCO).

        Returns:
            average_precision (np.array)
            precision (np.array)
            recall (np.array)
        """
        table = self.match_table[class_id].sort_values(by=["confidence"], ascending=False)
        matched_ind = {}
        nd = len(table)
        tp = np.zeros(nd, dtype=np.float64)
        fp = np.zeros(nd, dtype=np.float64)

        for d in range(nd):
            img_id, conf, iou, difficult, crowd, order = row_to_vars(table.iloc[d])
            if img_id not in matched_ind:
                matched_ind[img_id] = []
            res, idx = check_box(iou, difficult, crowd, order, matched_ind[img_id], iou_threshold, mpolicy)
            if res == "tp":
                tp[d] = 1
                matched_ind[img_id].append(idx)
            elif res == "fp":
                fp[d] = 1
        precision, recall = compute_precision_recall(tp, fp, self.class_statistics[:, class_id].sum())

        if recall_thresholds is None:
            average_precision = compute_average_precision(precision, recall)
        else:
            average_precision = compute_average_precision_with_recall_thresholds(precision, recall, recall_thresholds)

        return average_precision, precision, recall
