import json

import cv2


class Statistic(dict):
    def __str__(self) -> str:
        names = [k for k in self.keys()]
        max_name_length = max(len(str(name)) for name in names)
        template = "{:>" + str(max_name_length) + "}: {}"
        table_str = ""
        for name in names:
            table_str += template.format(name, str(self.__getitem__(name)))
            if name != names[-1]:
                table_str += "\n"
        return table_str


def clip(values, min_val=0.0, max_val=1.0):
    """Clip values of a sequence to [min_val, max_val] range.

    Args:
        values (Sequence[number]): numbers to clip
        min_value (number): minimum value of clip range
        max_value (number): maximam value of clip range

    Returns:
        List[number] with clipped numbers.
    """
    if isinstance(values, (int, float)):
        return min(max(values, min_val), max_val)
    return [min(max(num, min_val), max_val) for num in values]


def load_coco_json(path):
    """Read json with annotations.

    Args:
        path (str): path to .json file

    Raises:
        RuntimeError if .json file has no images
        RuntimeError if .json file has no categories

    Returns:
        images mapping - mapping from `image_id` to image annotations (`file_name`, `height`, `width`, `annotations`)
        categories mapping - mapping from `category_id` to `category_name`
    """

    with open(path, "r") as in_file:
        content = json.load(in_file)

    if not len(content["images"]):
        raise RuntimeError(f"There is no image records in '{path}' file!")

    if not len(content["categories"]):
        raise RuntimeError(f"There is no categories in '{path}' file!")

    images = {}  # image_id -> {file_name, height, width, annotations([{id, iscrowd, category_id, bbox}, ...])}
    for record in content["images"]:
        images[record["id"]] = {
            "file_name": record["file_name"],
            "height": record["height"],
            "width": record["width"],
            "annotations": [],
        }

    categories = {}  # category_id -> name
    for record in content["categories"]:
        categories[record["id"]] = record["name"]

    for record in content["annotations"]:
        images[record["image_id"]]["annotations"].append(
            {
                "id": record["id"],
                "iscrowd": record["iscrowd"],
                "category_id": record["category_id"],
                "bbox": record["bbox"],
            }
        )

    return images, categories


def read_image(path):
    """Read image from given path.

    Args:
        path (str): path to an image.

    Raises:
        FileNotFoundError when missing image file

    Returns:
        image (np.ndarray with dtype np.uint8)
    """
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"There is no '{path}'!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pixels_to_absolute(box, width, height):
    """Convert pixel coordinates to absolute scales ([0,1]).

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x, y, w, h).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x1, y1, x2, y2).
    """
    # type: () -> List[float]  # noqa: F821
    x, y, w, h = box
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def absolute_to_pixels(box, width, height):
    """Convert absolute coordinates to pixel values.

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x1, y1, x2, y2).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x, y, w, h).
    """
    x1, y1, x2, y2 = box
    return [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]
