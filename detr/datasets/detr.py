import os
import collections

import torch
from torch.utils.data import Dataset

from detr import utils


class DETRFileDataset(Dataset):
    def __init__(self, file, img_dir=None, transforms=None):
        """
        Args:
            file (str or pathlib.Path): path to a json file with annotations.
            img_dir (str): path to directory with images.
            transforms (albumentations.BasicTransform): transforms apply to images and bounding boxes.
                If `None` then images will be converted torch.Tensor (image will be divided by 255.
                and will be changed order to a [C, W, H]).
                Default is `None`.
        """
        self.file = file
        self.img_dir = img_dir
        self.transforms = transforms

        self.images, self.categories = utils.data.load_coco_json(file)
        self.images_list = sorted(self.images.keys())

        # mapping: class - category id
        self.class_to_cid = {}
        for cls_idx, cat_id in enumerate(sorted(self.categories.keys())):
            self.class_to_cid[cls_idx] = cat_id

        # mapping: category id - class
        self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}

        self.num_classes = len(self.class_to_cid)
        self.class_labels = [self.categories[self.class_to_cid[cls_idx]] for cls_idx in range(len(self.class_to_cid))]
        self.class_ids = [self.class_to_cid[class_number] for class_number in range(len(self.class_to_cid))]

        self.dataset_stats = None

    def __len__(self):
        return len(self.images_list)

    def get_class_mapping(self):
        """Information about class mapping.

        Returns:
            List with information about classes (List[Dict[str, Union[int, str]]]).
        """
        classes = [
            {
                "class": class_idx,
                "id": category_id,
                "name": self.categories[category_id],
            }
            for class_idx, category_id in self.class_to_cid.items()
        ]
        return classes

    def info(self):
        """Information about dataset.

        Returns:
            str with information about dataset
        """
        if self.dataset_stats is None:
            self.dataset_stats = utils.data.Statistic()
            self.dataset_stats["Number of images"] = len(self.images)
            self.dataset_stats["Number of labels"] = len(self.categories)
            self.dataset_stats["Labels"] = self.categories
            self.dataset_stats["Images with bounding boxes"] = sum(
                1 for img in self.images.values() if len(img["annotations"]) != 0
            )
            self.dataset_stats["Images without bounding boxes"] = sum(
                1 for img in self.images.values() if len(img["annotations"]) == 0
            )
            self.dataset_stats["Max number of objects on one image"] = max(
                len(img["annotations"]) for img in self.images.values()
            )
            self.dataset_stats["Min number of objects on one image"] = min(
                len(img["annotations"]) for img in self.images.values()
            )
            self.dataset_stats["Bounding box statistic"] = dict(
                collections.Counter(
                    self.categories[annot["category_id"]]
                    for img in self.images.values()
                    for annot in img["annotations"]
                ).most_common()
            )

        return str(self.dataset_stats)

    def __getitem__(self, index):
        # type: (int) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor] # noqa: F821
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.img_dir is not None:
            path = os.path.join(self.img_dir, path)
        image = utils.data.read_image(path)

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        for annotation in img_record["annotations"]:
            xyxy = utils.data.pixels_to_absolute(annotation["bbox"], img_record["width"], img_record["height"])
            xyxy = utils.data.clip(xyxy, 0.0, 1.0)
            assert all(0 <= num <= 1 for num in xyxy), f"All numbers should be in range [0, 1], but got {xyxy}!"
            bbox_class = str(self.cid_to_class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]
        else:
            image = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        labels_tensor = torch.zeros((len(boxes),), dtype=torch.long)
        boxes_tensor = torch.zeros((len(boxes), 4), dtype=torch.float)
        for i, (x1, y1, x2, y2, class_str) in enumerate(boxes):
            labels_tensor[i] = int(class_str)
            boxes_tensor[i, 0] = x1
            boxes_tensor[i, 1] = y1
            boxes_tensor[i, 2] = x2
            boxes_tensor[i, 3] = y2

        # (x1, y1, x2, y2) -> (x_center, y_center, w, h)
        boxes_tensor = utils.box_ops.change_box_order(boxes_tensor, "xyxy2xywh")

        return image, boxes_tensor, labels_tensor

    @staticmethod
    def collate_fn(batch):
        """Collect batch for DETR format.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]]):
                List with records from DETRFileDataset.

        Returns:
            NestedTensor with images and image masks
            List[Dict[str, torch.Tensor]] list with targets (bounding boxes and labels)
        """
        images = torch.stack([image for image, _, _ in batch], dim=0)
        masks = torch.zeros((images.size(0), images.size(2), images.size(3)), dtype=torch.bool)

        targets = [{"boxes": boxes, "labels": labels} for _, boxes, labels in batch]
        return utils.tensor.NestedTensor(images, masks), targets
