from .rlbench import RLBenchTrainTester


def fetch_train_tester(dataset_name):
    dataset_name = dataset_name.lower()
    if 'peract2' in dataset_name:
        return RLBenchTrainTester
    if 'rlbench' in dataset_name:
        return RLBenchTrainTester
    return None
