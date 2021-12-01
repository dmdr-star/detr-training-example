import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from detr.datasets.detr import DETRFileDataset
from detr.utils.misc import seed_all


def get_augmentations(img_size, img_mean, img_std, data_type=None, intensity=None):
    """Augmentations for training and validation datasets.

    Args:
        img_size (Tuple[int, int]): image sizes to use, expected pair of values - [width, height].
        img_mean (Tuple[float, float, float]): channelvise mean pixels value, channels order - [R, G, B].
        img_std (Tuple[float, float, float]): channelvise std value, channels order - [R, G, B].
        data_type (str): dataset type, should be `"training"` or `"validation"`.
            If `None` then will be used validation approach.
            Default is `None`.
        intensity (str): augmentations intensity, should be `"no_augmentations"` or `"basic"`
            or `"hard"` or `"hard_no_rotations"`.
            If `None` then will be used no augmentations approach.

    Returns:
        augmentations (albu.Compose) for a dataset.
    """
    if data_type is None:
        data_type = "validation"
    data_type = data_type.strip().lower()
    if data_type not in ("training", "validation"):
        raise RuntimeError(f"Unknown data_type ('{data_type}')! Expected 'training' or 'validation'!")

    if intensity is None:
        intensity = "no_augmentations"
    intensity = intensity.strip().lower()
    if intensity not in ("no_augmentations", "basic", "hard", "hard_no_rotations"):
        raise RuntimeError(f"Unknown load_type ('{intensity}')! Expected 'no_augmentations' or 'basic' or 'hard'!")

    if data_type == "validation":
        augmentations = albu.Compose(
            [
                albu.Resize(*img_size),
                albu.Normalize(mean=img_mean, std=img_std),
                ToTensorV2(),
            ],
            # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
            bbox_params=albu.BboxParams(format="albumentations"),
        )
    else:
        if intensity == "no_augmentations":
            augmentations = albu.Compose(
                [
                    albu.Resize(*img_size),
                    albu.Normalize(mean=img_mean, std=img_std),
                    ToTensorV2(),
                ],
                # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
                bbox_params=albu.BboxParams(format="albumentations"),
            )
        elif intensity == "basic":
            augmentations = albu.Compose(
                [
                    albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=10, border_mode=0, p=0.5),
                    albu.Resize(*img_size),
                    albu.Normalize(mean=img_mean, std=img_std),
                    albu.HorizontalFlip(p=0.1),
                    ToTensorV2(),
                ],
                # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
                bbox_params=albu.BboxParams("albumentations"),
            )
        elif intensity == "hard":
            augmentations = albu.Compose(
                [
                    albu.OneOf(
                        [
                            albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
                            albu.RandomGamma(),
                            albu.CLAHE(),
                        ]
                    ),
                    albu.RandomBrightnessContrast(brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
                    albu.OneOf(
                        [albu.Blur(), albu.MotionBlur(), albu.GaussNoise(), albu.ImageCompression(quality_lower=75)]
                    ),
                    albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=10, border_mode=0, p=0.5),
                    albu.Resize(*img_size),
                    albu.Normalize(mean=img_mean, std=img_std),
                    # lanscape/portrait mode on a phone
                    albu.OneOf(
                        [
                            albu.Rotate(limit=(90, 90)),
                            albu.Rotate(limit=(-90, -90)),
                        ],
                        p=0.5,
                    ),
                    albu.HorizontalFlip(p=0.1),
                    ToTensorV2(),
                ],
                # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
                bbox_params=albu.BboxParams("albumentations"),
            )
        elif intensity == "hard_no_rotations":
            augmentations = albu.Compose(
                [
                    albu.OneOf(
                        [
                            albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
                            albu.RandomGamma(),
                            albu.CLAHE(),
                        ]
                    ),
                    albu.RandomBrightnessContrast(brightness_limit=[-0.3, 0.3], contrast_limit=[-0.3, 0.3], p=0.5),
                    albu.OneOf(
                        [albu.Blur(), albu.MotionBlur(), albu.GaussNoise(), albu.ImageCompression(quality_lower=75)]
                    ),
                    albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=10, border_mode=0, p=0.5),
                    albu.Resize(*img_size),
                    albu.Normalize(mean=img_mean, std=img_std),
                    albu.HorizontalFlip(p=0.1),
                    ToTensorV2(),
                ],
                # use "albumentations" format because x1, y1, x2, y2 in range [0, 1]
                bbox_params=albu.BboxParams("albumentations"),
            )

    return augmentations


def get_loaders(
    train_annotations,
    train_images_dir,
    train_batch_size,
    train_num_workers,
    #
    valid_annotations,
    valid_images_dir,
    valid_batch_size,
    valid_num_workers,
    #
    image_size,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
    augmentations_intensity="basic",
):
    """Build training and validation loaders.

    Args:
        train_annotations (str or pathlib.Path): path to training dataset file.
        train_images_dir (str or pathlib.Path): path to directory with training images.
        train_batch_size (int): training batch size.
        train_num_workers (int): number of workers to use for creating training batches.
        valid_annotations (str or pathlib.Path): path to validation dataset file.
        valid_images_dir (str or pathlib.Path): path to directory with validation images.
        valid_batch_size (int): validation batch size.
        valid_num_workers (int): number of workers to use for creating validation batches.
        image_size (Tuple[int, int]): image sizes
        image_mean (Tuple[float, float, float]): channelvise mean pixels value
        image_std (Tuple[float, float, float]): channelvise std value
        augmentations_intensity (str): augmentation intensity, will be applied only to a training dataset.
            Expected one of `"no_augmentations"` or `"basic"` or `"hard"`.
            Default is `"basic"`.

    Raises:
        ValueError when passed wrong dataset type.

    Returns:
        train dataloader (torch.utils.data.DataLoader) and valid dataloader (torch.utils.data.DataLoader)
    """
    train_augmentations = get_augmentations(
        image_size, image_mean, image_std, data_type="training", intensity=augmentations_intensity
    )
    valid_augmentations = get_augmentations(image_size, image_mean, image_std, data_type="validation")

    train_dataset = DETRFileDataset(train_annotations, train_images_dir, transforms=train_augmentations)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=DETRFileDataset.collate_fn,
        worker_init_fn=seed_all,
    )

    valid_dataset = DETRFileDataset(valid_annotations, valid_images_dir, transforms=valid_augmentations)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=valid_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=DETRFileDataset.collate_fn,
    )
    return train_loader, valid_loader