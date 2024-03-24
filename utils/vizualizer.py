import pickle

from PIL import Image

from matplotlib import pyplot as plt
import numpy as nd

from detectron2.utils.visualizer import Visualizer


def read_image_to_numpy(image_path):
    img = Image.open(image_path)
    return nd.array(img)


def show(image):
    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def draw_mask(coco, image_id):
    image = read_image_to_numpy(coco.imgs[image_id]['file_name'])
    v = Visualizer(image)
    anns = coco.imgToAnns[image_id]
    masks = [coco.annToMask(ann) for ann in anns]
    colors = [color_map(ann['category_id']) for ann in anns]
    for mask, ann, color in zip(masks, anns, colors):
        v.draw_binary_mask(mask, color=color, alpha=0.0)
    return v


def draw_detection(coco_eval_, id_, type_='detection'):
    coco_main = coco_eval_.cocoGt if type_ == 'detection' else coco_eval_.cocoDt
    coco_aux = coco_eval_.cocoGt if type_ != 'detection' else coco_eval_.cocoDt
    text = 'DT' if type_ == 'detection' else 'GT'

    image_id = coco_main.anns[id_]['image_id']

    m_ann = coco_main.anns[id_]
    m_mask = coco_main.annToMask(m_ann)

    a_anns = coco_aux.imgToAnns[image_id]
    a_masks = [coco_aux.annToMask(gt_ann) for gt_ann in a_anns]
    colors = [color_map(ann['category_id']) for ann in a_anns]

    image_filename = coco_main.imgs[image_id]['file_name']
    print(image_id, '\t', image_filename)

    image = read_image_to_numpy(image_filename)
    v = Visualizer(image)
    v.draw_binary_mask(m_mask, color='black', edge_color='black', text=text, alpha=0.0)
    for mask, color in zip(a_masks, colors):
        v.draw_binary_mask(mask, color=color, alpha=0.0)

    show(v.get_output().get_image())


def draw_detections(coco_eval_, image_id):
    coco_gt = coco_eval_.cocoGt
    coco_dt = coco_eval_.cocoDt

    gt_anns = coco_gt.imgToAnns[image_id]
    gt_masks = [coco_gt.annToMask(ann) for ann in gt_anns]
    gt_colors = [color_map(ann['category_id']) for ann in gt_anns]
    gt_texts = ['GT:' + str(ann['id']) for ann in gt_anns]

    dt_anns = coco_dt.imgToAnns[image_id]
    dt_masks = [coco_dt.annToMask(ann) for ann in dt_anns]
    dt_colors = [color_map(ann['category_id']) for ann in dt_anns]
    dt_texts = [str(ann['id']) for ann in dt_anns]

    if image_id in coco_gt.imgs:
        image_filename = coco_gt.imgs[image_id]['file_name']
    else:
        image_filename = coco_dt.imgs[image_id]['file_name']

    print(image_id, '\t', image_filename)

    image = read_image_to_numpy(image_filename)
    v = Visualizer(image)
    for mask, color, text in zip(gt_masks, gt_colors, gt_texts):
        v.draw_binary_mask(mask, color=color, text=text, edge_color='black', alpha=0.0)
        # v.draw_binary_mask(mask, color=color, edge_color='black', alpha=0.0)
    for mask, color, text in zip(dt_masks, dt_colors, dt_texts):
        v.draw_binary_mask(mask, color=color, text=text, alpha=0.0)

    show(v.get_output().get_image())


def color_map(cat_id):
    label_color_map = [
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128)
    ]
    return {
        0: 'green',
        1: 'blue',
        2: 'red',
        3: 'CornflowerBlue',
    }[cat_id]


