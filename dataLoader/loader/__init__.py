
from .general_dataset import GeneralDataset


def get_dataset(name):
    return {
        'demo': GeneralDataset
    }[name]


