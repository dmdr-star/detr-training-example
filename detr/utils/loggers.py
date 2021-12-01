import json
import os
import logging
from datetime import datetime

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JSONLogger:
    def __init__(self, path, overwrite=True):
        """
        Args:
            path (str or python.Path): path to json
            overwrite (bool): option to overwite existing file with logs.
        """
        self.path = path
        self.overwrite = overwrite
        if os.path.isfile(self.path) and self.overwrite:
            os.remove(self.path)

    def __repr__(self):
        """Object representation."""
        return f"JSONLogger(path={self.path}, overwrite={self.overwrite})"

    def log(self, metrics, step, timestamp=None):
        """
        Args:
            metrics (Dict[str, Any]): metrics to save
            step (int): step number
            timestamp (float): unix timestamp, if `None` then will be used current time.
                Default is `None`.
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        # create empty file
        if not os.path.isfile(self.path):
            file_metrics = {}
        else:
            with open(self.path, "r") as f:
                file_metrics = json.load(f)

        # add new information
        for key, value in metrics.items():
            if key not in file_metrics:
                file_metrics[key] = []
            file_metrics[key].append({"step": step, "timestamp": timestamp, "value": value})

        # write back
        with open(self.path, "w") as f:
            json.dump(file_metrics, f, indent=2, cls=NumpyEncoder)


def log_metrics(logger, metrics, step, title=None, loader=None):
    """Log metrics.

    Args:
        logger (logging.Logger or alto_ai.loggers.JSONLogger): logger
        metrics (Dict[str, Any]): metrics
        step (int): step/epoch
        title (str): metrics title
        loader (str): loader name
    """
    if isinstance(logger, JSONLogger):
        if loader is not None:
            dump_metrics = {f"{loader}_{key}": value for key, value in metrics.items()}
        else:
            dump_metrics = metrics
        logger.log(metrics=dump_metrics, step=step)
    elif isinstance(logger, logging.Logger):
        metric_strings = []
        if title is not None:
            metric_strings.append(title)
        for key, value in metrics.items():
            if key == "confusion_matrix":
                _value = "\n" + "\n".join(("  " + " ".join(str(col) for col in row)) for row in value["matrix"])
            else:
                _value = value
            metric_strings.append(f" {key}: {_value}")
        if logger is not None:
            logger.info("\n".join(metric_strings))