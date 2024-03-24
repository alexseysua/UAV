import pickle
from pycocotools import mask
import copy

ioy_thr = 0.5
ioa_thr = 0.7


def get_statistics(coco_eval_, type_):
    assert type_ in ['gt', 'dt']
    coco_main = coco_eval_.cocoGt if type_ == 'gt' else coco_eval_.cocoDt
    coco_aux = coco_eval_.cocoDt if type_ == 'gt' else coco_eval_.cocoGt

    for i_ann in coco_main.anns.values():
        i_ann['match_best'] = None
        i_ann['matches_other'] = []

        o_anns = [i for i in coco_aux.anns.values() if i['image_id'] == i_ann['image_id']]
        if not o_anns:
            continue

        i_seg = i_ann['segmentation']
        o_segs = [o_ann['segmentation'] for o_ann in o_anns]
        o_iscrowds = [o_ann['iscrowd'] for o_ann in o_anns]

        i_area = i_ann['area']
        o_areas = [o_ann['area'] for o_ann in o_anns]
        intersect_areas = [mask.area(mask.merge([i_seg, o_seg], intersect=True)) for o_seg in o_segs]

        ious = mask.iou([i_seg], o_segs, o_iscrowds)[0]
        i_ioas = [i / i_area for i in intersect_areas]
        o_ioas = [i / j for i, j in zip(intersect_areas, o_areas)]

        # кастомный критерий: или iou > ioy_thr (это штатная часть)
        # или пересечение i_seg и o_seg входит в i_seg или o_seg (т.е. один какой-либо из сегментов
        # перекрывается другим) с порогом ioa_thr
        criteras = [i > ioy_thr or j > ioa_thr or k > ioa_thr for i, j, k in zip(ious, i_ioas, o_ioas)]

        # метрики по всем out, где criteria = True
        # разделяем метрики по max_iou (match_best) и остальные (matches_other)
        metrics = [[*i] for i in zip(o_anns, ious, i_ioas, o_ioas)]
        metrics = [i for i, j in zip(metrics, criteras) if j]
        metrics.sort(key=lambda x: x[1], reverse=True)

        if metrics:
            i_ann['match_best'] = copy.deepcopy(metrics[0])
            i_ann['matches_other'] = [copy.deepcopy(i) for i in metrics[1:]]

        if not i_ann['match_best']:
            i_ann['status'] = 'FN' if type_ == 'gt' else 'FP'
        else:
            category_match = i_ann['category_id'] == i_ann['match_best'][0]['category_id']
            i_ann['status'] = 'TP' if category_match else 'WC'


def print_statistics():
    def process(ann):
        best = ann['match_best']
        if best:
            print(ann['id'], best[0]['id'], ann['category_id'], best[0]['category_id'],
                  ann['image_id'], *best[1:], ann['status'], sep='\t')
        else:
            print(ann['id'], "", ann['category_id'], "", ann['image_id'], "", "", "", ann['status'], sep='\t')

    for x in coco_eval.cocoGt.anns.values():
        process(x)
    print("")
    for x in coco_eval.cocoDt.anns.values():
        process(x)


with open('C:/Files/Projects/python/UAV/scripts/detectron/coco_eval.pickle', 'rb') as file:
    coco_eval = pickle.load(file)

get_statistics(coco_eval, 'gt')
get_statistics(coco_eval, 'dt')
print_statistics()
