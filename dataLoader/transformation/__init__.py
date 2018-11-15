
# 加载数据集进行变换预处理的的接口
from .transform_demo import driver_transform


# 把写好的transform借口注册到字典中
transformations = dict({
    'driver': driver_transform,
})


def get_dataset_transformation(transform_name):
    support_transform_names = transformations.keys()
    if transform_name not in support_transform_names:
        raise ValueError("Unsupported transform:{}, and we just support {}. "
                         "Do you forget to register your transformation?"
                         .format(transform_name, list(support_transform_names)))
    else:
        return support_transform_names[transform_name]


