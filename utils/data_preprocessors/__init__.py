from functools import partial

from .rlbench import RLBenchDataPreprocessor


def fetch_data_preprocessor(dataset_name, img_size=256):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=img_size)
    if 'rlbench' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=img_size)
    return None
