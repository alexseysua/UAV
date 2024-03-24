import os.path

root = os.path.dirname(__file__)
models = os.path.join(root, 'models')
runs = os.path.join(root, 'runs')
weights = os.path.join(root, 'weights')

datasets_path = os.path.join(root, 'datasets')
datasets = {
    'Aerial_Cars_v3': os.path.join(datasets_path, 'Aerial_Cars_v3')
}


# def get_data_yaml(dataset):
#     ds = os.path.join(datasets[dataset], "data.yaml")
#     return ds



