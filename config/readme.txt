


整个模型的参数都在这里定义
支持的加载参数的方式包括：
    1.json文件--(Optic-disc和Attention-gated都是用的这个)
    2.argparse
    3.txt文件  retina-unet用的这个

TODO: 写一个这些配置文件的转换工具，都转换成  对象.参数


目前需要加载的参数
    (1)数据预处理---
        is_divide_to_patch

        首先指定数据集名称、
        数据集是否是图片，如果是图片是否分成patches训练。
        1.是否分patches
          TODO 注意：如果分patch 意味着会把处理成二进制npz/hdf5格式的数据一次性加载进内存。比较吃内存。
          TODO 可以尝试分patch后，将得到的数据集分割成多个二进制文件。但是这样又有不能随机加载数据集的问题！
          是：{
            patch_height,
            patch_width,
            patch_nums_on_image,
            save_type:hdf5 or numpy,    # 这两种本质上都是numpy.ndarray
            save_path:已经指定
          }

        2.最终传递给








