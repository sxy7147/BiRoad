from .rlbench import RLBenchDepth2Cloud


def fetch_depth2cloud(dataset_name, img_size=256):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchDepth2Cloud((img_size, img_size))
    if 'rlbench' in dataset_name:
        return RLBenchDepth2Cloud((img_size, img_size))
    return None
