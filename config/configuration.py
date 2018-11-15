
from .utils import *


class Configuration:
    def __init__(self, config_file):
        self.config_file = config_file
        if self.config_file is None:
            print("No Config file.")




