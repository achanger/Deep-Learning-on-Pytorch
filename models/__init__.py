
# load networks interface. 加载网络模型的接口
# Step 1. Import your defined network. 把自己定义的网络加载进来
from networks.segmentation.basicUnet import U_Net


# Step 2. Register your network by using dict: key:model_name | value: NetName
# 把自己的网络通过字典关联起来, 后面通多传递model_name获取该模型
networks = dict({
    'unet': U_Net,
})


def get_model(model_name):
    support_model_names = networks.keys()
    if model_name not in support_model_names:
        raise ValueError("Unsupported model:{}, and we just support {}. Do you forget to register your networks?"
                         .format(model_name, list(support_model_names)))
    else:
        return networks[model_name]





