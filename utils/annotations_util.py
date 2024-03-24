import json
import os
import shutil
from pathlib import Path


def filter_by_categories(path, categories_id, process_files=True):
    """
    Фильтрует файл разметки в формате COCO по заданным категориям включая перечень файлов в соответствии
    с параметром files
    :param path: путь до файла с аннотациями
    :param categories_id: list категорий (ids), которые нужно оставить
    :param process_files: если True, то перечень файлов также фильтруется
    """

    categories_id = categories_id if isinstance(categories_id, list) else [categories_id]
    file = Path(path)
    data = json.loads(file.read_text())

    if process_files:
        images_id = [i['image_id'] for i in data['annotations'] if i['category_id'] in categories_id]
        data['images'] = [i for i in data['images'] if i['id'] in images_id]
    data['categories'] = [i for i in data['categories'] if i['id'] in categories_id]
    data['annotations'] = [i for i in data['annotations'] if i['category_id'] in categories_id]

    new_file = file.with_stem(file.stem + '_changed')
    new_file.write_text(json.dumps(data, indent=4))


def change_labels(src, dest=None):
    """
    Меняет label всех ID классов на 0 в фале с разметкой в формате YOLO
    с параметром files
    :param src: Директория, где лежат исходные файлы с разметкой в формате YOLO
    :param dest: Директория, куда будут сложены откорректированные файлы. Если None, то: "{srs}_converted"
    """

    dest = dest or src + "_converted"
    Path(dest).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(src):
        with open(os.path.join(src, file)) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = '0' + lines[i][1:]
        with open(os.path.join(dest, file), 'w') as f:
            f.writelines(lines)


def segs_to_bboxes(labels_path, yolo_style=True):
    """
    Конвертирует файлы из labels_path с разметкой YOLO segments в bboxes. Новые файлы сохраняются
    в "{labels_path}_converted". Если {labels_path} уже существует, то он удаляется вместе со всем содержимым
    bbox сохраняется в форматах: (X_Center, Y_Center, W, H) или (X_min, Y_min, X_max, Y_max),
    в зависимости от значения параметра yolo_style
    :param labels_path: Путь, где лежат исходные файлы
    :param yolo_style: if True output format is YOLO Style (X_Center, Y_Center, W, H) else (X_min, Y_min, X_max, Y_max)
    """

    def seg_to_bbox(seg_info):
        """
        :param seg_info: Example input: 5 0.04687 0.36914 0.064453 0.38476 0.080078 0.40234 ...
        """
        class_id, *points = seg_info.split()
        points = [float(p) for p in points]
        x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
        if not yolo_style:
            bbox_info = f"{int(class_id)} {x_min} {y_min} {x_max} {y_max}"
        else:
            width, height = x_max - x_min, y_max - y_min
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            bbox_info = f"{int(class_id)} {x_center} {y_center} {width} {height}"
        return bbox_info

    files = [f for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]
    labels_converted_path = labels_path + '_converted'

    # Если 'labels_converted_path' уже существует, то удаляем его
    if os.path.isdir(labels_converted_path):
        shutil.rmtree(labels_converted_path)
    os.mkdir(labels_converted_path)

    for file in files:
        with open(os.path.join(labels_path, file)) as f:
            segments = f.readlines()
            segments_converted = [seg_to_bbox(s) for s in segments]
            segments_converted[:-1] = [s + '\n' for s in segments_converted[:-1]]
        with open(os.path.join(labels_converted_path, file), 'w') as f:
            f.writelines(segments_converted)
