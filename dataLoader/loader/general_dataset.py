import os
import torch.utils.data as data


class GeneralDataset(data.Dataset):
    def __init__(self, config, mode, transform=None, preload_data=False):
        super(GeneralDataset, self).__init__()

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Loading is done\n')


    def __getitem__(self, index):
        if self.transform:
            self.input, self.target = self.transform(self.input, self.target)
        return self.input, self.target

    def __len__(self):
        return len(self.image_filenames)

