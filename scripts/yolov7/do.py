import os

import paths

model = f'{paths.models}\\yolov7'

# yolo params
data = f'{paths.datasets["Aerial_Cars_v3"]}\\data.yaml'
weights = f'{paths.weights}\\yolov7\\standard\\yolov7-tiny.pt'
hyp = '{model}\\data\\hyp.scratch.p5.yaml'

img_size = 416
project_path = paths.runs
project_name = 'yolo_aerial_cars'


# TRAIN
cmd = (f'python "{model}\\train.py " '
       f'--img-size {img_size} '
       f'--data {data} '
       f'--batch 16 '
       f'--epochs 100 '
       f'--weights {weights} '
       f'--hyp "{hyp}" '
       f'--workers 24 '
       f'--project {project_path} '
       f'--name {project_name}'
       )
print(cmd)
os.system(cmd)

# TEST
# os.system(f"python test.py --weights {weights} --source \"{images}\" --save-txt")
# os.system(f"python test.py --img-size {img_size} --data {data} --weights {weights}")
