import os
import random

import cv2

import detectron2.model_zoo
import paths
from demo.predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator, LVISEvaluator
from detectron2.utils.visualizer import Visualizer, ColorMode

from scripts.detectron.utils import register_dataset, split_dataset_and_register_result, cv2_imshow, increment_path
from scripts.detectron.variables import Avtotransport, Cotlovan

data = Cotlovan

config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
output_path = 'detectron/output'

output_path = os.path.join(paths.runs, output_path)
output_path = increment_path(output_path)


def prepare_dataset(data_, split_ratio=0.9):
    DatasetCatalog.clear()  # Очистить зарегистрированные датасеты
    register_dataset(data_)
    return split_dataset_and_register_result(data_, split_ratio)


def set_config():
    cfg_ = get_cfg()
    cfg_.merge_from_file(detectron2.model_zoo.model_zoo.get_config_file(config_path))

    cfg_.DATASETS.TRAIN = (data.dataset_name + "_train",)
    cfg_.DATASETS.TEST = (data.dataset_name + "_val",)
    cfg_.DATALOADER.NUM_WORKERS = 0
    cfg_.MODEL.WEIGHTS = detectron2.model_zoo.model_zoo.get_checkpoint_url(config_path)  # Let training initialize

    # from model zoo
    cfg_.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
    cfg_.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg_.SOLVER.MAX_ITER = 1000  # 300 iterations seems good enough for this toy dataset; you will need to train
    # longer for a practical dataset
    cfg_.SOLVER.STEPS = []  # do not decay learning rate
    cfg_.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg_.MODEL.ROI_HEADS.NUM_CLASSES = 4  # NOTE: this config means the number of classes, but a few popular unofficial
    # tutorials incorrect uses num_classes+1 here.
    cfg_.MODEL.DEVICE = 'cpu'
    cfg_.OUTPUT_DIR = output_path

    return cfg_


def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def test():
    # weights_dir = os.path.join(paths.runs, output_path + '4', "model_final.pth")
    weights_dir = "C:/Files/Projects/python/UAV/runs/detectron/cotlovans/model_final.pth"
    cfg.MODEL.WEIGHTS = weights_dir  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(dataset_val_metadata.name, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False,)
    val_loader = build_detection_test_loader(cfg, dataset_val_metadata.name)

    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(inference)


def demo():
    weights_dir = os.path.join(paths.runs, output_path + '4', "model_final.pth")
    cfg.MODEL.WEIGHTS = weights_dir  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshol
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(os.path.join(data.images_path, "34_Avtotransport_bf8.jpg"))
    # cv2_imshow(im)
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


dataset_train_metadata, dataset_train_dicts, dataset_val_metadata, dataset_val_dicts = \
    prepare_dataset(data, split_ratio=0)
cfg = set_config()

test()



