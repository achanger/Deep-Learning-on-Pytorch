
from .configuration import Configuration


def get_config(config_file=None):
    return Configuration(config_file)


x = dict({
    "name": "achange",
})

print(repr(x))



