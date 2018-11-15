
# 加载数据集的接口
from .transformation import get_dataset_transformation
from torch.utils.data import DataLoader, sampler


def get_dataloader(config, mode=None):
    if mode is None:
        raise ValueError("Type what dataset to load, Train or Valid or Test?")

    return DataLoader(dataset=config.train_dataset, num_workers=config.num_workers,
                      batch_size=config.batch_size, sampler=config.train_sampler)



