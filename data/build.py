from data import datasets

def build_dataset(cfg):
    dataset_factory = getattr(datasets, cfg.DATASET.FACTORY)
    dataset = dataset_factory(cfg)
    return dataset.batch(cfg.DATASET.BATCH_SIZE)
