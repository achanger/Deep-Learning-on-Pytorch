"""
    transforms的实例。
    【输入】*inputs必须是张量或者是图像的路径(包含文件名)！！
           如果输入是多张图像路径/多个张量，将他们依次传入driver_transform()即可，注意最后一个参数是mode...
    【输出】是最终喂给模型的张量(numpy.ndarray/torch.tensor/tensorflow.placeholder等等)。


   【注意】在做分割的时候，彩色图像和分割图像需要做同步的处理！
          特别是在含有随机变换的时候！！

          对图像的预处理本质上是做矩阵运算。
          如：pytorch的torchvision.transforms本质上是通过numpy/Pillow进行变换的。

    下面就以比较复杂的处理DRIVER数据集为例，演示具体的数据预处理：
"""

import torch
import random
import numbers
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as F


class Crop(object):
    def __init__(self, parameters=None):
        self.params = parameters

    def __call__(self, img):
        i, j, h, w = self.params
        return F.crop(img, i, j, h, w)


class MyRandomCrop(object):
    def __init__(self, size, parameters=None, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.params = parameters

    @staticmethod
    def get_params(rawimg_size, output_size):
        w, h = rawimg_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return [i, j, th, tw]

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.params

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return F.hflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        return F.vflip(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


def _train_transform(*inputs):
    image, target = inputs
    print("np.asanyarray(image).shape", np.asanyarray(image).shape)
    print("np.asanyarray(target).shape", np.asanyarray(target).shape)

    transforms = []
    # 1. 先从图像中剪裁INPUT_IMAGE_SIZE大小 然后进行后面的数据增强
    params = MyRandomCrop.get_params(rawimg_size=image.size, output_size=(512,512))
    transforms.append(MyRandomCrop(size=(512, 512), parameters=params))

    # 2. 图像翻转
    if random.random() < 0.5:
        transforms.append(RandomHorizontalFlip())  # torchvision.transforms.functional.hflip(img)
    if random.random() < 0.5:
        transforms.append(RandomVerticalFlip())

    # 3.旋转图像： 不旋转、旋转90、180或270度
    RotationDegrees = [0, 90, 180, 270]
    RotationDegree = random.randint(0, 3)
    RotationDegree = RotationDegrees[RotationDegree]
    transforms.append(T.RandomRotation((RotationDegree, RotationDegree)))

    # image和GT做相同处理的部分
    transform = T.Compose(transforms)
    image = transform(image)
    target = transform(target)

    """以下做不同的处理： """
    # 4. 调整彩图的亮度、对比度、色调
    # 5. 对彩图进行标准化、并转换为torch.tensor
    ts = T.Compose([
        # T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = ts(image)

    # 6. 对GT图像进行二值化、并转换为torch.tensor TODO: 注意需要添加一个维度包起来, 和input对应

    target = torch.from_numpy(np.asanyarray(target, dtype=np.uint8) / 255).int()
    target = target.unsqueeze(0)

    return image, target


def _valid_transform(*inputs):
    return T.Compose([
             T.Scale(512),
             T.CenterCrop(512),
             T.ToTensor(),
             normalize,
        ])


def _test_transform(*inputs):
    return T.Compose([
             T.Scale(512),
             T.CenterCrop(512),
             T.ToTensor(),
             normalize,
        ])


transforms = dict({
    "train": _train_transform,
    "valid": _valid_transform,
    "test": _test_transform
})


def driver_transform(*inputs, mode="train"):
    assert mode in ['train', 'valid', 'test']
    assert type(inputs) == tuple

    x, y = inputs
    print(x, y)
    quit()
    # 在这里可以将inputs参数unpack
    transforms_function = transforms[mode]
    return transforms_function(*inputs)


if __name__ == '__main__':
    driver_transform(10, 20, mode="kkk")

