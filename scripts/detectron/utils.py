import re
import random

import cv2
import glob

from matplotlib import pyplot as plt
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_dataset(data):
    register_coco_instances(data.dataset_name, {}, data.labels_path, data.images_path)


def split_dataset_and_register_result(data, train_ratio):
    dataset = DatasetCatalog.get(data.dataset_name)

    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)
    train_size = int(train_ratio * data_size)

    hard_coded_train_indices = [i for i in indices[:train_size]]
    hard_coded_val_indices = [i for i in indices[train_size:]]
    # train_dataset = [dataset[i] for i in indices[:train_size]]
    # val_dataset = [dataset[i] for i in indices[train_size:]]
    train_dataset = [dataset[i] for i in hard_coded_train_indices]
    val_dataset = [dataset[i] for i in hard_coded_val_indices]

    print("> len train_dataset:", len(train_dataset))
    print("> len val_dataset:", len(val_dataset))

    train_dataset_name = data.dataset_name + "_train"
    val_dataset_name = data.dataset_name + "_val"

    DatasetCatalog.register(train_dataset_name, lambda: train_dataset)
    MetadataCatalog.get(train_dataset_name).set(
        thing_classes=MetadataCatalog.get(data.dataset_name).thing_classes)
    DatasetCatalog.register(val_dataset_name, lambda: val_dataset)
    MetadataCatalog.get(val_dataset_name).set(thing_classes=MetadataCatalog.get(data.dataset_name).thing_classes)

    dataset_train_metadata = MetadataCatalog.get(train_dataset_name)
    dataset_train_dicts = DatasetCatalog.get(train_dataset_name)
    dataset_val_metadata = MetadataCatalog.get(val_dataset_name)
    dataset_val_dicts = DatasetCatalog.get(val_dataset_name)

    return dataset_train_metadata, dataset_train_dicts, dataset_val_metadata, dataset_val_dicts


def cv2_imshow(image, figsize=(25, 23)):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)  #
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def increment_path(path, exist_ok=False, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path